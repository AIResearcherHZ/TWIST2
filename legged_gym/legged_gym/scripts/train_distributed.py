# Multi-GPU Distributed Training Script for TWIST2
# Uses PyTorch DDP for gradient synchronization across multiple GPUs

import os

import isaacgym  # noqa: F401 - must be imported before torch
from legged_gym.envs import *  # noqa: F401, F403
from legged_gym import LEGGED_GYM_ROOT_DIR, LEGGED_GYM_ENVS_DIR
from legged_gym.gym_utils import get_args, task_registry

import torch
import torch.distributed as dist
import wandb

def setup_distributed():
    """Initialize distributed training environment."""
    # Get distributed training info from environment variables (set by torchrun)
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    rank = int(os.environ.get("RANK", 0))
    
    # Set CUDA device for this process
    torch.cuda.set_device(local_rank)
    
    # Initialize process group
    if world_size > 1:
        dist.init_process_group(
            backend="nccl",
            init_method="env://",
            world_size=world_size,
            rank=rank
        )
    
    return local_rank, world_size, rank


def cleanup_distributed():
    """Clean up distributed training."""
    if dist.is_initialized():
        dist.destroy_process_group()


def is_main_process():
    """Check if this is the main process (rank 0)."""
    return int(os.environ.get("RANK", 0)) == 0


def train(args):
    # Setup distributed training
    local_rank, world_size, rank = setup_distributed()
    
    # Override device settings based on local rank
    args.device = f"cuda:{local_rank}"
    args.sim_device = f"cuda:{local_rank}"
    args.rl_device = f"cuda:{local_rank}"
    args.graphics_device_id = local_rank
    args.headless = True  # Always headless for multi-GPU
    
    # Adjust seed for each process to ensure different random states
    if args.seed is not None:
        args.seed = args.seed + rank
    
    log_pth = LEGGED_GYM_ROOT_DIR + "/logs/{}/".format(args.proj_name) + args.exptid
    
    # All processes create directories to avoid race condition
    os.makedirs(log_pth, exist_ok=True)
    
    # Synchronize after directory creation
    if world_size > 1:
        dist.barrier()
    
    # Only main process logs to wandb
    if is_main_process():
        
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
            
        # Extract robot type from task name (same logic as train.py)
        robot_type = args.task.split("_")[0]
        
        try:
            wandb.init(
                entity="far-wandb",
                project="twist",
                name=args.exptid,
                mode=mode,
                dir="../../logs"
            )
        except:
            wandb.init(
                project=args.proj_name,
                name=args.exptid,
                mode=mode,
                dir="../../logs"
            )
    else:
        # Non-main processes don't log to wandb
        args.no_wandb = True
    
    # Synchronize all processes before creating environments
    if world_size > 1:
        dist.barrier()
    
    if is_main_process():
        print(f"Creating environment on {args.device}...")
    env, _ = task_registry.make_env(name=args.task, args=args)
    if is_main_process():
        print(f"Using motion file: {env.cfg.motion.motion_file}")
    
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
    
    # Start training
    ppo_runner.learn(
        num_learning_iterations=train_cfg.runner.max_iterations, 
        init_at_random_ep_len=True
    )
    
    # Cleanup
    cleanup_distributed()


if __name__ == "__main__":
    args = get_args()
    train(args)