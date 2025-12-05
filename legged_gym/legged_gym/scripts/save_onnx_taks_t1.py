#!/usr/bin/env python3
"""
ONNX conversion script for taks_t1_stu_future (student policy with future motion)
Usage: python save_onnx_taks_t1.py --ckpt_path <absolute_path_to_checkpoint>
"""

import os
import sys
sys.path.append("../../../rsl_rl")
import torch
import torch.nn as nn
from rsl_rl.modules.actor_critic_future import ActorFuture, get_activation
import argparse
from termcolor import cprint


class HardwareStudentFutureNN(nn.Module):
    """Hardware deployment wrapper for student policy with future motion support."""

    def __init__(self,
                 num_observations,
                 num_motion_observations,
                 num_priop_observations,
                 num_motion_steps,
                 num_future_observations,
                 num_future_steps,
                 motion_latent_dim,
                 future_latent_dim,
                 num_actions,
                 actor_hidden_dims,
                 activation,
                 history_latent_dim,
                 num_history_steps,
                 layer_norm=False,
                 tanh_encoder_output=False,
                 **kwargs):
        super().__init__()

        self.num_observations = num_observations
        self.num_actions = num_actions
        self.num_motion_observations = num_motion_observations
        self.num_priop_observations = num_priop_observations

        activation = get_activation(activation)

        self.normalizer = None

        self.actor = ActorFuture(
            num_observations=num_observations,
            num_motion_observations=num_motion_observations,
            num_priop_observations=num_priop_observations,
            num_motion_steps=num_motion_steps,
            num_future_observations=num_future_observations,
            num_future_steps=num_future_steps,
            motion_latent_dim=motion_latent_dim,
            future_latent_dim=future_latent_dim,
            num_actions=num_actions,
            actor_hidden_dims=actor_hidden_dims,
            activation=activation,
            history_latent_dim=history_latent_dim,
            num_history_steps=num_history_steps,
            layer_norm=layer_norm,
            tanh_encoder_output=tanh_encoder_output,
            **kwargs
        )

    def load_normalizer(self, normalizer):
        self.normalizer = normalizer

    def forward(self, obs):
        assert obs.shape[1] == self.num_observations, \
            f"Expected {self.num_observations} but got {obs.shape[1]}"
        obs = self.normalizer.normalize(obs)
        return self.actor(obs)


def convert_to_onnx(args):
    """Convert taks_t1_stu_future student policy to ONNX."""

    ckpt_path = args.ckpt_path

    if not os.path.exists(ckpt_path):
        cprint(f"Error: Checkpoint file not found: {ckpt_path}", "red")
        return

    # Taks_T1 student future configuration
    robot_name = "taks_t1"
    num_actions = 32  # Taks_T1 has 32 DOF
    history_len = 10

    # Motion observations (current frame only for student)
    num_motion_steps = 1
    n_mimic_obs_single = 6 + 32  # 38 dims per frame
    num_motion_observations = num_motion_steps * n_mimic_obs_single

    # Proprioceptive observations
    num_priop_observations = 3 + 2 + 3 * num_actions  # 101

    # Future motion observations
    num_future_steps = 1  # len(tar_motion_steps_future)
    n_future_obs_single = 6 + 32  # 38 dims per frame
    num_future_observations = num_future_steps * n_future_obs_single

    # Single step observation size (for history)
    n_obs_single = num_motion_observations + num_priop_observations

    # Total observation size
    num_observations = n_obs_single * (history_len + 1) + num_future_observations

    # Network architecture parameters
    motion_latent_dim = 128
    future_latent_dim = 128
    history_latent_dim = 128
    actor_hidden_dims = [512, 512, 256, 128]
    activation = 'silu'

    print(f"Taks_T1 Student Future Policy Configuration:")
    print(f"  Robot: {robot_name}")
    print(f"  Actions: {num_actions}")
    print(f"  History length: {history_len}")
    print(f"  Motion observations: {num_motion_observations}")
    print(f"  Proprioceptive observations: {num_priop_observations}")
    print(f"  Future observations: {num_future_observations}")
    print(f"  Single obs size: {n_obs_single}")
    print(f"  Total observations: {num_observations}")
    print(f"  Motion latent dim: {motion_latent_dim}")
    print(f"  Future latent dim: {future_latent_dim}")
    print(f"  History latent dim: {history_latent_dim}")
    print("")

    device = torch.device('cuda')
    policy = HardwareStudentFutureNN(
        num_observations=num_observations,
        num_motion_observations=num_motion_observations,
        num_priop_observations=num_priop_observations,
        num_motion_steps=num_motion_steps,
        num_future_observations=num_future_observations,
        num_future_steps=num_future_steps,
        motion_latent_dim=motion_latent_dim,
        future_latent_dim=future_latent_dim,
        num_actions=num_actions,
        actor_hidden_dims=actor_hidden_dims,
        activation=activation,
        history_latent_dim=history_latent_dim,
        num_history_steps=history_len,
        layer_norm=True,
        tanh_encoder_output=False,
        use_history_encoder=True,
        use_motion_encoder=True
    ).to(device)

    cprint(f"Loading model from: {ckpt_path}", "green")

    ac_state_dict = torch.load(ckpt_path, map_location=device, weights_only=False)
    policy.load_state_dict(ac_state_dict['model_state_dict'], strict=False)
    policy.load_normalizer(ac_state_dict['normalizer'])

    policy = policy.to(device)

    policy.eval()
    with torch.no_grad():
        batch_size = 1
        obs_input = torch.ones(batch_size, num_observations, device=device)
        cprint(f"Input observation shape: {obs_input.shape}", "cyan")

        onnx_path = ckpt_path.replace('.pt', '.onnx')

        torch.onnx.export(
            policy,
            obs_input,
            onnx_path,
            export_params=True,
            opset_version=11,
            do_constant_folding=True,
            input_names=['input'],
            output_names=['output'],
            dynamic_axes={
                'input': {0: 'batch_size'},
                'output': {0: 'batch_size'}
            }
        )
        cprint(f"ONNX model saved to: {onnx_path}", "green")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Convert taks_t1_stu_future student policy to ONNX')
    parser.add_argument('--ckpt_path', type=str, required=True,
                        help='Absolute path to checkpoint file')
    args = parser.parse_args()
    convert_to_onnx(args)
