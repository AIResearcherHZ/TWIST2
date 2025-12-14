# Multi-GPU Distributed Training Script for TWIST2
# Uses PyTorch DDP for gradient synchronization across multiple GPUs
# Optimized for efficient multi-GPU training with:
#   - Efficient gradient synchronization (single all_reduce)
#   - Mixed precision training (AMP)
#   - Proper barrier synchronization
#   - NCCL optimizations

import os

# NCCL optimizations for better multi-GPU performance
os.environ.setdefault("NCCL_IB_DISABLE", "1")  # Disable InfiniBand if not available
os.environ.setdefault("NCCL_P2P_LEVEL", "NVL")  # Use NVLink if available
os.environ.setdefault("NCCL_ASYNC_ERROR_HANDLING", "1")  # Better error handling

import isaacgym  # noqa: F401 - must be imported before torch
from legged_gym.envs import *  # noqa: F401, F403
from legged_gym import LEGGED_GYM_ROOT_DIR, LEGGED_GYM_ENVS_DIR
from legged_gym.gym_utils import get_args, task_registry

import torch
import torch.distributed as dist
import wandb
import time


def setup_distributed():
    """Initialize distributed training environment with optimizations."""
    # Get distributed training info from environment variables (set by torchrun)
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    rank = int(os.environ.get("RANK", 0))
    
    # Set CUDA device for this process BEFORE any CUDA operations
    torch.cuda.set_device(local_rank)
    
    # Enable TF32 for better performance on Ampere GPUs
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    
    # Enable cudnn benchmark for consistent input sizes
    torch.backends.cudnn.benchmark = True
    
    # Initialize process group with timeout for robustness
    if world_size > 1:
        dist.init_process_group(
            backend="nccl",
            init_method="env://",
            world_size=world_size,
            rank=rank,
            timeout=torch.distributed.default_pg_timeout
        )
        
        # Verify initialization
        if rank == 0:
            print(f"[Distributed] Initialized with {world_size} GPUs")
            print(f"[Distributed] NCCL version: {torch.cuda.nccl.version()}")
    
    return local_rank, world_size, rank


def cleanup_distributed():
    """Clean up distributed training."""
    if dist.is_initialized():
        dist.destroy_process_group()


def is_main_process():
    """Check if this is the main process (rank 0)."""
    return int(os.environ.get("RANK", 0)) == 0


def print_rank0(msg, rank=None):
    """Print only from rank 0."""
    if rank is None:
        rank = int(os.environ.get("RANK", 0))
    if rank == 0:
        print(msg)


def train(args):
    start_time = time.time()
    
    # Setup distributed training
    local_rank, world_size, rank = setup_distributed()
    
    # Override device settings based on local rank
    args.device = f"cuda:{local_rank}"
    args.sim_device = f"cuda:{local_rank}"
    args.rl_device = f"cuda:{local_rank}"
    args.graphics_device_id = local_rank
    args.headless = True  # Always headless for multi-GPU
    
    # Adjust seed for each process to ensure different random states
    # Use a larger offset to ensure truly different sequences
    if args.seed is not None:
        args.seed = args.seed + rank * 1000
    
    log_pth = LEGGED_GYM_ROOT_DIR + "/logs/{}/".format(args.proj_name) + args.exptid
    
    # Only main process creates directories and logs to wandb
    if is_main_process():
        try:
            os.makedirs(log_pth, exist_ok=True)
        except Exception as e:
            print(f"Warning: Could not create log directory: {e}")
        
        if args.debug:
            mode = "disabled"
            args.rows = 10
            args.cols = 5
            args.num_envs = 4
            args.headless = False
        else:
            mode = "online"
        
        if args.no_wandb:
            mode = "disabled"
            
        robot_type = args.task.split("_")[0]
        
        # Calculate effective batch size
        total_envs = args.num_envs * world_size
        
        try:
            wandb.init(
                entity="far-wandb", 
                project="twist", 
                name=f"{args.exptid}_multigpu_{world_size}",
                mode=mode, 
                dir="../../logs",
                config={
                    "world_size": world_size,
                    "num_envs_per_gpu": args.num_envs,
                    "total_envs": total_envs,
                    "distributed": True,
                    "amp_enabled": True,
                }
            )
        except:
            wandb.init(
                project="g1_mimic", 
                name=f"{args.exptid}_multigpu_{world_size}",
                mode=mode, 
                dir="../../logs"
            )
        
        if robot_type == "g1":
            wandb.save(LEGGED_GYM_ENVS_DIR + "/g1/g1_mimic_distill_config.py", policy="now")
        
        print("="*60)
        print(f"Distributed Training Configuration")
        print("="*60)
        print(f"  World Size: {world_size}")
        print(f"  Envs per GPU: {args.num_envs}")
        print(f"  Total Envs: {total_envs}")
        print(f"  AMP Enabled: True")
        print("="*60)
    else:
        # Non-main processes don't log to wandb
        args.no_wandb = True
    
    # Synchronize all processes before creating environments
    # This ensures all processes are ready before heavy GPU operations
    if world_size > 1:
        dist.barrier()
        if rank == 0:
            print("[Distributed] All processes synchronized, creating environments...")
    
    print(f"[Rank {rank}] Creating environment on {args.device}...")
    env, _ = task_registry.make_env(name=args.task, args=args)
    print(f"[Rank {rank}] Environment created. Motion file: {env.cfg.motion.motion_file}")
    
    # Synchronize after environment creation
    if world_size > 1:
        dist.barrier()
        if rank == 0:
            print("[Distributed] All environments created, starting training...")
    
    # Create runner with distributed training support
    ppo_runner, train_cfg = task_registry.make_alg_runner(
        log_root=log_pth, 
        env=env, 
        name=args.task, 
        args=args,
        distributed=True,
        world_size=world_size,
        rank=rank,
        local_rank=local_rank
    )
    
    setup_time = time.time() - start_time
    print_rank0(f"[Distributed] Setup completed in {setup_time:.2f}s", rank)
    
    # Start training
    try:
        ppo_runner.learn(
            num_learning_iterations=train_cfg.runner.max_iterations, 
            init_at_random_ep_len=True
        )
    except KeyboardInterrupt:
        print_rank0("\n[Distributed] Training interrupted by user", rank)
    except Exception as e:
        print(f"[Rank {rank}] Error during training: {e}")
        raise
    finally:
        # Cleanup
        cleanup_distributed()
        print_rank0(f"[Distributed] Training finished. Total time: {time.time() - start_time:.2f}s", rank)


if __name__ == "__main__":
    args = get_args()
    train(args)
