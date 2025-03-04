import argparse
import glob
import os
from typing import Union

import ale_py
import cv2
import gym
import gymnasium as gym
import numpy as np
import shimmy
import torch
import torch.nn as nn
import tqdm
from gymnasium.wrappers import FrameStackObservation
from stable_baselines3.common.atari_wrappers import (
    EpisodicLifeEnv,
    FireResetEnv,
    MaxAndSkipEnv,
    NoopResetEnv,
    WarpFrame,
)
from stable_baselines3.common.buffers import ReplayBuffer
from torch import Tensor
from torchsummary import summary


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Argument parser for training an agent to play atari games"
    )
    parser.add_argument(
        "--env_id",
        help="id of registered gym environment. Raises error if gym env is not registered",
        type=str,
    )
    parser.add_argument(
        "--seed",
        help="seed for reproducible experiment. Defaults to 42 if not specified",
        type=int,
        default=42,
    )
    parser.add_argument(
        "--batch_size",
        help="data batch size for training q-network",
        type=int,
        default=64,
    )
    parser.add_argument(
        "--total_timesteps",
        help="maximum length of an episode. Defaults to total_timesteps of registered env",
        type=int,
    )
    parser.add_argument(
        "--epsilon",
        help="epsilon greedy for choosing action in agent env",
        type=int,
        default=0.1,
    )
    parser.add_argument("--device", help="GPU device", type=str, default="cpu")
    parser.add_argument(
        "--buffer_size", help="experience replay memory size", type=int, default=10_000
    )
    parser.add_argument(
        "--total_episodes", help="total number of episodes", type=int, default=10_000
    )
    parser.add_argument(
        "--discount_factor",
        help="discount factor for cumulative rewards",
        type=int,
        default=0.99,
    )
    return parser.parse_args()


class DQN(nn.Module):
    def __init__(self, input_size: int, output_size: int):
        super().__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.network = nn.Sequential(
            nn.Conv2d(4, 16, 8, stride=4),
            nn.ReLU(),
            nn.Conv2d(16, 32, 4, stride=2),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(self._compute_dim(), self.output_size),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.network(x / 255.0)

    def _compute_dim(self) -> int:
        conv1_output_size = (self.input_size - 8) // 4 + 1
        conv2_output_size = (conv1_output_size - 4) // 2 + 1
        return 32 * conv2_output_size**2


class Agent:
    def __init__(
        self,
        buffer_size: int,
        env_id: str,
        total_timesteps: int,
        device: Union[torch.device, str],
        epsilon: int,
    ):
        self.buffer_size = buffer_size
        self.env = self._wrap_env(self._create_env(env_id, total_timesteps))
        self.epsilon = epsilon
        self.device = device
        self.replay_buffer = self._init_replay_buffer(buffer_size)
        self.dqn = DQN(
            input_size=self.env.observation_space.shape[1],
            output_size=self.env.action_space.n,
        ).to(device)

    def _init_replay_buffer(self, buffer_size: int):
        replay_buffer = ReplayBuffer(
            buffer_size=buffer_size,
            observation_space=self.env.observation_space,
            action_space=self.env.action_space,
            device=self.device,
        )

        curr_size = 0
        while curr_size < self.buffer_size:
            curr_obs, info = self.env.reset()
            while True:
                action = self.env.action_space.sample()
                next_obs, reward, terminated, truncated, info = self.env.step(action)
                replay_buffer.add(
                    obs=curr_obs,
                    next_obs=next_obs,
                    action=action,
                    reward=reward,
                    done=terminated,
                    infos=[info],
                )
                if terminated or truncated:
                    break
                curr_size += 1
                curr_obs = next_obs
        return replay_buffer

    def _create_env(self, env_id: str, total_timesteps: int):
        return gym.make(
            id=env_id,
            render_mode="rgb_array",
            max_episode_steps=total_timesteps,
        )

    def _wrap_env(self, env):
        env = WarpFrame(env=env, width=84, height=84)
        env = MaxAndSkipEnv(env=env, skip=4)
        env = FrameStackObservation(env=env, stack_size=4)
        env = EpisodicLifeEnv(env=env)
        env = FireResetEnv(env=env)
        env = NoopResetEnv(env, noop_max=30)
        return env

    def _choose_action(self, state):
        state = torch.from_numpy(state).float().to(self.device)
        state = torch.squeeze(torch.unsqueeze(state, 0), 4)
        prob = np.random.rand()
        if prob < self.epsilon:
            action = self.env.action_space.sample()
        else:
            with torch.no_grad():
                self.dqn.eval()
                action = torch.argmax(self.dqn(state))
        return action.cpu().numpy() if isinstance(action, Tensor) else action

    def _get_max_q_value(self, state):
        state = torch.from_numpy(state).float().to(self.device)
        state = torch.squeeze(torch.unsqueeze(state, 0), 4)
        with torch.no_grad():
            self.dqn.eval()
            max_q_value = torch.max(self.dqn(state))
        return max_q_value.cpu().numpy()


def main(args):
    if args.device == "cuda":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    elif args.device == "mps":
        if not torch.backends.mps.is_available():
            if not torch.backends.mps.is_built():
                print(
                    "MPS not available because the current PyTorch install was not built with MPS enabled."
                )
            else:
                print(
                    "MPS not available because the current MacOS version is not 12.3+ and/or you do not have an MPS-enabled device on this machine."
                )

        else:
            device = torch.device("mps")

    agent = Agent(
        buffer_size=args.buffer_size,
        env_id=args.env_id,
        total_timesteps=args.total_timesteps,
        device=device,
        epsilon=args.epsilon,
    )

    for _ in range(args.total_episodes):
        curr_state, info = agent.env.reset()
        while True:
            action = agent._choose_action(curr_state)
            next_state, reward, terminated, truncated, info = agent.env.step(action)
            agent.replay_buffer.add(
                obs=curr_state,
                next_obs=next_state,
                action=action,
                reward=reward,
                done=terminated,
                infos=[info],
            )

            minibatch = agent.replay_buffer.sample(args.batch_size)
            states = torch.squeeze(minibatch[0], 4)
            targets = np.zeros((args.batch_size, 1))
            for i in range(args.batch_size):
                is_terminal = minibatch[3][i][0]
                reward = minibatch[4][i][0]
                if is_terminal:
                    targets[i] = reward.cpu().numpy()
                else:
                    targets[
                        i
                    ] = reward.cpu().numpy() + args.discount_factor * agent._get_max_q_value(
                        next_state
                    )

            targets = torch.from_numpy(targets).float().to(device)

            agent.dqn.train()
            for data, target in zip(states, targets):
                breakpoint()

            curr_state = next_state
            if terminated or truncated:
                break


if __name__ == "__main__":
    args = parse_arguments()
    main(args)
