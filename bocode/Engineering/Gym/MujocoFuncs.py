import gymnasium as gym
import torch
from typing import Tuple
from ...base import *

class AntProblem(BenchmarkProblem):
    available_dimensions = 8
    input_type = DataType.CONTINUOUS
    num_objectives = 1
    num_constraints = 0

    def __init__(self):
        super().__init__(
            dim=self.__class__.available_dimensions,
            num_objectives=self.__class__.num_objectives,
            num_constraints=0,
            bounds=list(zip(
                gym.make('Ant-v5').action_space.low.tolist(),
                gym.make('Ant-v5').action_space.high.tolist()
            ))
        )
        self.env = gym.make('Ant-v5')

    def _evaluate_implementation(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        batch_size = x.shape[0]
        rewards = torch.zeros(batch_size, self.__class__.num_objectives)
        for i in range(batch_size):
            obs, _ = self.env.reset()
            action = x[i].cpu().numpy()
            obs, reward, done, truncated, info = self.env.step(action)
            rewards[i, 0] = -reward
        return None, rewards


class HalfCheetahProblem(BenchmarkProblem):
    available_dimensions = 6
    input_type = DataType.CONTINUOUS
    num_objectives = 1
    num_constraints = 0

    def __init__(self):
        super().__init__(
            dim=self.__class__.available_dimensions,
            num_objectives=self.__class__.num_objectives,
            num_constraints=0,
            bounds=list(zip(
                gym.make('HalfCheetah-v5').action_space.low.tolist(),
                gym.make('HalfCheetah-v5').action_space.high.tolist()
            ))
        )
        self.env = gym.make('HalfCheetah-v5')

    def _evaluate_implementation(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        batch_size = x.shape[0]
        rewards = torch.zeros(batch_size, self.__class__.num_objectives)
        for i in range(batch_size):
            obs, _ = self.env.reset()
            action = x[i].cpu().numpy()
            obs, reward, done, truncated, info = self.env.step(action)
            rewards[i, 0] = -reward
        return None, rewards


class HopperProblem(BenchmarkProblem):
    available_dimensions = 3
    input_type = DataType.CONTINUOUS
    num_objectives = 1
    num_constraints = 0

    def __init__(self):
        super().__init__(
            dim=self.__class__.available_dimensions,
            num_objectives=self.__class__.num_objectives,
            num_constraints=0,
            bounds=list(zip(
                gym.make('Hopper-v5').action_space.low.tolist(),
                gym.make('Hopper-v5').action_space.high.tolist()
            ))
        )
        self.env = gym.make('Hopper-v5')

    def _evaluate_implementation(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        batch_size = x.shape[0]
        rewards = torch.zeros(batch_size, self.__class__.num_objectives)
        for i in range(batch_size):
            obs, _ = self.env.reset()
            action = x[i].cpu().numpy()
            obs, reward, done, truncated, info = self.env.step(action)
            rewards[i, 0] = -reward
        return None, rewards


class HumanoidProblem(BenchmarkProblem):
    available_dimensions = 17
    input_type = DataType.CONTINUOUS
    num_objectives = 1
    num_constraints = 0

    def __init__(self):
        super().__init__(
            dim=self.__class__.available_dimensions,
            num_objectives=self.__class__.num_objectives,
            num_constraints=0,
            bounds=list(zip(
                gym.make('Humanoid-v5').action_space.low.tolist(),
                gym.make('Humanoid-v5').action_space.high.tolist()
            ))
        )
        self.env = gym.make('Humanoid-v5')

    def _evaluate_implementation(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        batch_size = x.shape[0]
        rewards = torch.zeros(batch_size, self.__class__.num_objectives)
        for i in range(batch_size):
            obs, _ = self.env.reset()
            action = x[i].cpu().numpy()
            obs, reward, done, truncated, info = self.env.step(action)
            rewards[i, 0] = -reward
        return None, rewards


class HumanoidStandupProblem(BenchmarkProblem):
    available_dimensions = 17
    input_type = DataType.CONTINUOUS
    num_objectives = 1
    num_constraints = 0

    def __init__(self):
        super().__init__(
            dim=self.__class__.available_dimensions,
            num_objectives=self.__class__.num_objectives,
            num_constraints=0,
            bounds=list(zip(
                gym.make('HumanoidStandup-v5').action_space.low.tolist(),
                gym.make('HumanoidStandup-v5').action_space.high.tolist()
            ))
        )
        self.env = gym.make('HumanoidStandup-v5')

    def _evaluate_implementation(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        batch_size = x.shape[0]
        rewards = torch.zeros(batch_size, self.__class__.num_objectives)
        for i in range(batch_size):
            obs, _ = self.env.reset()
            action = x[i].cpu().numpy()
            obs, reward, done, truncated, info = self.env.step(action)
            rewards[i, 0] = -reward
        return None, rewards


class InvertedDoublePendulumProblem(BenchmarkProblem):
    available_dimensions = 1
    input_type = DataType.CONTINUOUS
    num_objectives = 1
    num_constraints = 0

    def __init__(self):
        super().__init__(
            dim=self.__class__.available_dimensions,
            num_objectives=self.__class__.num_objectives,
            num_constraints=0,
            bounds=list(zip(
                gym.make('InvertedDoublePendulum-v5').action_space.low.tolist(),
                gym.make('InvertedDoublePendulum-v5').action_space.high.tolist()
            ))
        )
        self.env = gym.make('InvertedDoublePendulum-v5')

    def _evaluate_implementation(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        batch_size = x.shape[0]
        rewards = torch.zeros(batch_size, self.__class__.num_objectives)
        for i in range(batch_size):
            obs, _ = self.env.reset()
            action = x[i].cpu().numpy()
            obs, reward, done, truncated, info = self.env.step(action)
            rewards[i, 0] = -reward
        return None, rewards


class InvertedPendulumProblem(BenchmarkProblem):
    available_dimensions = 1
    input_type = DataType.CONTINUOUS
    num_objectives = 1
    num_constraints = 0

    def __init__(self):
        super().__init__(
            dim=self.__class__.available_dimensions,
            num_objectives=self.__class__.num_objectives,
            num_constraints=0,
            bounds=list(zip(
                gym.make('InvertedPendulum-v5').action_space.low.tolist(),
                gym.make('InvertedPendulum-v5').action_space.high.tolist()
            ))
        )
        self.env = gym.make('InvertedPendulum-v5')

    def _evaluate_implementation(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        batch_size = x.shape[0]
        rewards = torch.zeros(batch_size, self.__class__.num_objectives)
        for i in range(batch_size):
            obs, _ = self.env.reset()
            action = x[i].cpu().numpy()
            obs, reward, done, truncated, info = self.env.step(action)
            rewards[i, 0] = -reward
        return None, rewards


class PusherProblem(BenchmarkProblem):
    available_dimensions = 7
    input_type = DataType.CONTINUOUS
    num_objectives = 1
    num_constraints = 0

    def __init__(self):
        super().__init__(
            dim=self.__class__.available_dimensions,
            num_objectives=self.__class__.num_objectives,
            num_constraints=0,
            bounds=list(zip(
                gym.make('Pusher-v5').action_space.low.tolist(),
                gym.make('Pusher-v5').action_space.high.tolist()
            ))
        )
        self.env = gym.make('Pusher-v5')

    def _evaluate_implementation(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        batch_size = x.shape[0]
        rewards = torch.zeros(batch_size, self.__class__.num_objectives)
        for i in range(batch_size):
            obs, _ = self.env.reset()
            action = x[i].cpu().numpy()
            obs, reward, done, truncated, info = self.env.step(action)
            rewards[i, 0] = -reward
        return None, rewards


class ReacherProblem(BenchmarkProblem):
    available_dimensions = 2
    input_type = DataType.CONTINUOUS
    num_objectives = 1
    num_constraints = 0

    def __init__(self):
        super().__init__(
            dim=self.__class__.available_dimensions,
            num_objectives=self.__class__.num_objectives,
            num_constraints=0,
            bounds=list(zip(
                gym.make('Reacher-v5').action_space.low.tolist(),
                gym.make('Reacher-v5').action_space.high.tolist()
            ))
        )
        self.env = gym.make('Reacher-v5')

    def _evaluate_implementation(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        batch_size = x.shape[0]
        rewards = torch.zeros(batch_size, self.__class__.num_objectives)
        for i in range(batch_size):
            obs, _ = self.env.reset()
            action = x[i].cpu().numpy()
            obs, reward, done, truncated, info = self.env.step(action)
            rewards[i, 0] = -reward
        return None, rewards


class SwimmerProblem(BenchmarkProblem):
    available_dimensions = 2
    input_type = DataType.CONTINUOUS
    num_objectives = 1
    num_constraints = 0

    def __init__(self):
        super().__init__(
            dim=self.__class__.available_dimensions,
            num_objectives=self.__class__.num_objectives,
            num_constraints=0,
            bounds=list(zip(
                gym.make('Swimmer-v5').action_space.low.tolist(),
                gym.make('Swimmer-v5').action_space.high.tolist()
            ))
        )
        self.env = gym.make('Swimmer-v5')

    def _evaluate_implementation(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        batch_size = x.shape[0]
        rewards = torch.zeros(batch_size, self.__class__.num_objectives)
        for i in range(batch_size):
            obs, _ = self.env.reset()
            action = x[i].cpu().numpy()
            obs, reward, done, truncated, info = self.env.step(action)
            rewards[i, 0] = -reward
        return None, rewards


class Walker2DProblem(BenchmarkProblem):
    available_dimensions = 6
    input_type = DataType.CONTINUOUS
    num_objectives = 1
    num_constraints = 0

    def __init__(self):
        super().__init__(
            dim=self.__class__.available_dimensions,
            num_objectives=self.__class__.num_objectives,
            num_constraints=0,
            bounds=list(zip(
                gym.make('Walker2d-v5').action_space.low.tolist(),
                gym.make('Walker2d-v5').action_space.high.tolist()
            ))
        )
        self.env = gym.make('Walker2d-v5')

    def _evaluate_implementation(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        batch_size = x.shape[0]
        rewards = torch.zeros(batch_size, self.__class__.num_objectives)
        for i in range(batch_size):
            obs, _ = self.env.reset()
            action = x[i].cpu().numpy()
            obs, reward, done, truncated, info = self.env.step(action)
            rewards[i, 0] = -reward
        return None, rewards
