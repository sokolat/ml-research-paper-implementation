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
from gymnasium.wrappers import FrameStack, RecordEpisodeStatistics
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
from tqdm import tqdm


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
        default=0.2,
    )
    parser.add_argument("--device", help="GPU device", type=str, default="cpu")
    parser.add_argument(
        "--buffer_size", help="experience replay memory size", type=int, default=10_000
    )
    parser.add_argument(
        "--gamma",
        help="discount factor for cumulative rewards",
        type=float,
        default=0.99,
    )
    parser.add_argument(
        "--learning_rate",
        help="learning rate for gradient descent",
        type=float,
        default=0.1,
    )
    parser.add_argument(
        "--num_envs", help="number of environment to create", type=int, default=1
    )
    parser.add_argument(
        "--update_freq",
        help="number of iteration before updating the netowrk",
        type=int,
        default=1000,
    )
    parser.add_argument(
        "--epsilon_min",
        help="minimum epsilon value for exploration",
        type=int,
        default=0.01,
    )
    parser.add_argument(
        "--epsilon_max",
        help="maximum epsilon value for exploration",
        type=int,
        default=1.0,
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
        env_id: str,
        total_timesteps: int,
        device: Union[torch.device, str],
        num_envs: int,
        seed: int,
    ):
        self.seed = seed
        self.envs = self._create_envs(env_id, total_timesteps, num_envs)
        self.device = device
        self.q_network = DQN(
            input_size=self.envs.observation_space.shape[2],
            output_size=self.envs.action_space[0].n,
        ).to(device)
        self.target_network = DQN(
            input_size=self.envs.observation_space.shape[2],
            output_size=self.envs.action_space[0].n,
        ).to(device)

    def _create_envs(self, env_id: str, total_timesteps: int, num_envs: int):
        def call_back():
            env = gym.make(
                id=env_id,
                max_episode_steps=total_timesteps,
            )
            env = NoopResetEnv(env, noop_max=30)
            env = MaxAndSkipEnv(env=env, skip=4)
            env = EpisodicLifeEnv(env=env)
            env = RecordEpisodeStatistics(env)
            env = FireResetEnv(env=env)
            env = WarpFrame(env=env, width=84, height=84)
            env = FrameStack(env=env, num_stack=4)

            env.action_space.seed(self.seed)
            return env

        envs = gym.vector.SyncVectorEnv([call_back for _ in range(num_envs)])
        return envs


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
        env_id=args.env_id,
        total_timesteps=args.total_timesteps,
        device=device,
        num_envs=args.num_envs,
        seed=args.seed,
    )

    def get_max_q_value():
        with torch.no_grad():
            agent.target_network.eval()
            max_q_value = torch.max(agent.target_network(states))
        return max_q_value

    def linear_epsilon_decay():
        return max(
            args.epsilon_min,
            args.epsilon_max
            - (step / args.total_timesteps) * (args.epsilon_max - args.epsilon_min),
        )

    curr_states, _ = agent.envs.reset(seed=args.seed)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(
        params=agent.q_network.parameters(), lr=args.learning_rate
    )

    update_counter = 0

    agent.target_network.load_state_dict(agent.q_network.state_dict())

    replay_buffer = ReplayBuffer(
        buffer_size=args.buffer_size,
        observation_space=agent.envs.observation_space,
        action_space=agent.envs.action_space,
        device=args.device,
    )

    curr_obs, _ = agent.envs.reset(seed=args.seed)
    for _ in range(args.buffer_size):
        actions = agent.envs.action_space.sample()
        next_obs, rewards, terminated, truncated, infos = agent.envs.step(actions)
        replay_buffer.add(
            obs=curr_obs,
            next_obs=next_obs,
            action=actions,
            reward=rewards,
            done=terminated,
            infos=[infos],
        )
        curr_obs = next_obs

    for step in tqdm(range(args.total_timesteps)):
        states = torch.from_numpy(curr_states).float().to(args.device)
        states = torch.squeeze(states, 4)
        prob = np.random.rand()
        epsilon = linear_epsilon_decay()
        if prob < epsilon:
            actions = agent.envs.action_space.sample()
        else:
            with torch.no_grad():
                agent.q_network.eval()
                actions = torch.argmax(agent.q_network(states), dim=1)

        actions = actions.cpu().numpy() if isinstance(actions, Tensor) else actions

        next_states, rewards, terminated, truncated, infos = agent.envs.step(actions)
        breakpoint()

        replay_buffer.add(
            obs=curr_states,
            next_obs=next_states,
            action=actions,
            reward=rewards,
            done=terminated,
            infos=[infos],
        )

        minibatch = replay_buffer.sample(args.batch_size)
        states = torch.squeeze(torch.squeeze(minibatch.observations, 1), 4)
        actions = minibatch.actions.long()
        dones = minibatch.dones
        rewards = minibatch.rewards

        targets = rewards + torch.where(dones == 1, 0, args.gamma * get_max_q_value())

        agent.q_network.train()
        output = agent.q_network(states)
        q_values = torch.gather(output, 1, actions)

        loss = criterion(q_values, targets)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        curr_states = next_states

        update_counter += 1
        if update_counter % args.update_freq == 0:
            agent.target_network.load_state_dict(agent.q_network.state_dict())


if __name__ == "__main__":
    args = parse_arguments()
    main(args)
