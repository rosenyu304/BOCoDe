from pathlib import Path
from typing import Optional, Tuple

import gymnasium as gym
import numpy as np
import torch

from ...base import BenchmarkProblem, DataType


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
            bounds=list(
                zip(
                    gym.make("Ant-v5").action_space.low.tolist(),
                    gym.make("Ant-v5").action_space.high.tolist(),
                )
            ),
        )
        self.env = gym.make("Ant-v5")

    def _evaluate_implementation(
        self, x: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
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
            bounds=list(
                zip(
                    gym.make("HalfCheetah-v5").action_space.low.tolist(),
                    gym.make("HalfCheetah-v5").action_space.high.tolist(),
                )
            ),
        )
        self.env = gym.make("HalfCheetah-v5")

    def _evaluate_implementation(
        self, x: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
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
            bounds=list(
                zip(
                    gym.make("Hopper-v5").action_space.low.tolist(),
                    gym.make("Hopper-v5").action_space.high.tolist(),
                )
            ),
        )
        self.env = gym.make("Hopper-v5")

    def _evaluate_implementation(
        self, x: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
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
            bounds=list(
                zip(
                    gym.make("Humanoid-v5").action_space.low.tolist(),
                    gym.make("Humanoid-v5").action_space.high.tolist(),
                )
            ),
        )
        self.env = gym.make("Humanoid-v5")

    def _evaluate_implementation(
        self, x: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
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
            bounds=list(
                zip(
                    gym.make("HumanoidStandup-v5").action_space.low.tolist(),
                    gym.make("HumanoidStandup-v5").action_space.high.tolist(),
                )
            ),
        )
        self.env = gym.make("HumanoidStandup-v5")

    def _evaluate_implementation(
        self, x: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
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
            bounds=list(
                zip(
                    gym.make("InvertedDoublePendulum-v5").action_space.low.tolist(),
                    gym.make("InvertedDoublePendulum-v5").action_space.high.tolist(),
                )
            ),
        )
        self.env = gym.make("InvertedDoublePendulum-v5")

    def _evaluate_implementation(
        self, x: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
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
            bounds=list(
                zip(
                    gym.make("InvertedPendulum-v5").action_space.low.tolist(),
                    gym.make("InvertedPendulum-v5").action_space.high.tolist(),
                )
            ),
        )
        self.env = gym.make("InvertedPendulum-v5")

    def _evaluate_implementation(
        self, x: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
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
            bounds=list(
                zip(
                    gym.make("Pusher-v5").action_space.low.tolist(),
                    gym.make("Pusher-v5").action_space.high.tolist(),
                )
            ),
        )
        self.env = gym.make("Pusher-v5")

    def _evaluate_implementation(
        self, x: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
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
            bounds=list(
                zip(
                    gym.make("Reacher-v5").action_space.low.tolist(),
                    gym.make("Reacher-v5").action_space.high.tolist(),
                )
            ),
        )
        self.env = gym.make("Reacher-v5")

    def _evaluate_implementation(
        self, x: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
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
            bounds=list(
                zip(
                    gym.make("Walker2d-v5").action_space.low.tolist(),
                    gym.make("Walker2d-v5").action_space.high.tolist(),
                )
            ),
        )
        self.env = gym.make("Walker2d-v5")

    def _evaluate_implementation(
        self, x: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
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
            bounds=list(
                zip(
                    gym.make("Swimmer-v5").action_space.low.tolist(),
                    gym.make("Swimmer-v5").action_space.high.tolist(),
                )
            ),
        )
        self.env = gym.make("Swimmer-v5")

    def _evaluate_implementation(
        self, x: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        batch_size = x.shape[0]
        rewards = torch.zeros(batch_size, self.__class__.num_objectives)
        for i in range(batch_size):
            obs, _ = self.env.reset()
            action = x[i].cpu().numpy()
            obs, reward, done, truncated, info = self.env.step(action)
            rewards[i, 0] = -reward
        return None, rewards


class SwimmerPolicySearchProblem(BenchmarkProblem):
    available_dimensions = 16  # <-- matches D above
    input_type = DataType.CONTINUOUS
    num_objectives = 1
    num_constraints = 0

    def __init__(
        self,
        num_rollouts: int = 5,
        render: bool = False,
        template_file: Optional[str] = None,
    ):
        # ---------------------------------------------------------------------
        self.env = gym.make("Swimmer-v5")
        self.num_rollouts = num_rollouts
        self.render = render
        script_dir = Path(__file__).parent
        template_file = (
            script_dir / "mujoco_policies" / "Swimmer-v1" / "lin_policy_plus.npz"
        )

        # -------- template policy, mean, std ---------------------------------
        if template_file is not None:
            arr_0 = np.load(template_file, allow_pickle=True)["arr_0"]
            self.W_shape = arr_0[0].shape  # (2,8)
            self.obs_mean = arr_0[1]
            self.obs_std = arr_0[2]
        else:
            self.W_shape = (
                self.env.action_space.shape[0],
                self.env.observation_space.shape[0],
            )
            self.obs_mean = np.zeros(self.W_shape[1])
            self.obs_std = np.ones(self.W_shape[1])

        dim = int(np.prod(self.W_shape))  # 16

        # ---- element-wise bounds on the weights (same as LA-MCTS code) ------
        lb, ub = -1.0, 1.0  # for Swimmer in MujucoPolicyFunc
        bounds = [(lb, ub)] * dim

        super().__init__(
            dim=dim,
            num_objectives=self.__class__.num_objectives,
            num_constraints=0,
            bounds=bounds,
        )

    # -------------------------------------------------------------------------
    def _evaluate_implementation(
        self, x: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        batch_size = x.shape[0]
        fvals = torch.empty(batch_size, 1, device=x.device)

        for i in range(batch_size):
            # reshape flat vector back to a (2,8) weight matrix
            W = x[i].detach().cpu().numpy().reshape(self.W_shape)

            # average return over N roll-outs (like the Facebook code)
            total_return = 0.0
            for _ in range(self.num_rollouts):
                obs, _ = self.env.reset()
                done = truncated = False
                episode_return = 0.0

                while not (done or truncated):
                    # linear state-feedback control
                    action = np.dot(W, (obs - self.obs_mean) / self.obs_std)
                    # keep the action in the legal torque range
                    action = np.clip(
                        action, self.env.action_space.low, self.env.action_space.high
                    )

                    obs, reward, done, truncated, _ = self.env.step(action)
                    episode_return += reward
                    if self.render:
                        self.env.render()

                total_return += episode_return

            fvals[i, 0] = -total_return / self.num_rollouts  # negate if you MINIMISE
        return None, fvals


class AntPolicySearchProblem(BenchmarkProblem):
    available_dimensions = 840  # <-- matches D above
    input_type = DataType.CONTINUOUS
    num_objectives = 1
    num_constraints = 0

    def __init__(
        self,
        num_rollouts: int = 5,
        render: bool = False,
        template_file: Optional[str] = None,
    ):
        # ---------------------------------------------------------------------
        self.env = gym.make("Ant-v5")
        self.num_rollouts = num_rollouts
        self.render = render
        script_dir = Path(__file__).parent
        template_file = (
            script_dir / "mujoco_policies" / "Ant-v1" / "lin_policy_plus.npz"
        )

        # -------- template policy, mean, std ---------------------------------
        if template_file is not None:
            arr_0 = np.load(template_file, allow_pickle=True)["arr_0"]
            self.W_shape = np.delete(arr_0[0], slice(27, 33), axis=1).shape
            self.obs_mean = np.delete(arr_0[1], slice(27, 33))
            self.obs_std = np.delete(arr_0[2], slice(27, 33))
        else:
            self.W_shape = (
                self.env.action_space.shape[0],
                self.env.observation_space.shape[0],
            )
            self.obs_mean = np.zeros(self.W_shape[1])
            self.obs_std = np.ones(self.W_shape[1])

        dim = int(np.prod(self.W_shape))

        # ---- element-wise bounds on the weights (same as LA-MCTS code) ------
        lb, ub = -1.0, 1.0
        bounds = [(lb, ub)] * dim

        super().__init__(
            dim=dim,
            num_objectives=self.__class__.num_objectives,
            num_constraints=0,
            bounds=bounds,
        )

    # -------------------------------------------------------------------------
    def _evaluate_implementation(
        self, x: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        batch_size = x.shape[0]
        fvals = torch.empty(batch_size, 1, device=x.device)

        for i in range(batch_size):
            # reshape flat vector back to a weight matrix
            W = x[i].detach().cpu().numpy().reshape(self.W_shape)

            # average return over N roll-outs (like the Facebook code)
            total_return = 0.0
            for _ in range(self.num_rollouts):
                obs, _ = self.env.reset()
                done = truncated = False
                episode_return = 0.0

                while not (done or truncated):
                    # linear state-feedback control
                    action = np.dot(W, (obs - self.obs_mean) / self.obs_std)
                    # keep the action in the legal torque range
                    action = np.clip(
                        action, self.env.action_space.low, self.env.action_space.high
                    )

                    obs, reward, done, truncated, _ = self.env.step(action)
                    episode_return += reward
                    if self.render:
                        self.env.render()

                total_return += episode_return

            fvals[i, 0] = total_return / self.num_rollouts  # negate if you MINIMISE
        return None, fvals


class HalfCheetahPolicySearchProblem(BenchmarkProblem):
    available_dimensions = 102  # <-- matches D above
    input_type = DataType.CONTINUOUS
    num_objectives = 1
    num_constraints = 0

    def __init__(
        self,
        num_rollouts: int = 5,
        render: bool = False,
        template_file: Optional[str] = None,
    ):
        # ---------------------------------------------------------------------
        self.env = gym.make("HalfCheetah-v5")
        self.num_rollouts = num_rollouts
        self.render = render
        script_dir = Path(__file__).parent
        template_file = (
            script_dir / "mujoco_policies" / "HalfCheetah-v1" / "lin_policy_plus.npz"
        )

        # -------- template policy, mean, std ---------------------------------
        if template_file is not None:
            arr_0 = np.load(template_file, allow_pickle=True)["arr_0"]
            self.W_shape = arr_0[0].shape  # (2,8)
            self.obs_mean = arr_0[1]
            self.obs_std = arr_0[2]
        else:
            self.W_shape = (
                self.env.action_space.shape[0],
                self.env.observation_space.shape[0],
            )
            self.obs_mean = np.zeros(self.W_shape[1])
            self.obs_std = np.ones(self.W_shape[1])

        dim = int(np.prod(self.W_shape))  # 16

        # ---- element-wise bounds on the weights (same as LA-MCTS code) ------
        lb, ub = -1.0, 1.0  # for Swimmer in MujucoPolicyFunc
        bounds = [(lb, ub)] * dim

        super().__init__(
            dim=dim,
            num_objectives=self.__class__.num_objectives,
            num_constraints=0,
            bounds=bounds,
        )

    # -------------------------------------------------------------------------
    def _evaluate_implementation(
        self, x: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        batch_size = x.shape[0]
        fvals = torch.empty(batch_size, 1, device=x.device)

        for i in range(batch_size):
            # reshape flat vector back to a (2,8) weight matrix
            W = x[i].detach().cpu().numpy().reshape(self.W_shape)

            # average return over N roll-outs (like the Facebook code)
            total_return = 0.0
            for _ in range(self.num_rollouts):
                obs, _ = self.env.reset()
                done = truncated = False
                episode_return = 0.0

                while not (done or truncated):
                    # linear state-feedback control
                    action = np.dot(W, (obs - self.obs_mean) / self.obs_std)
                    # keep the action in the legal torque range
                    action = np.clip(
                        action, self.env.action_space.low, self.env.action_space.high
                    )

                    obs, reward, done, truncated, _ = self.env.step(action)
                    episode_return += reward
                    if self.render:
                        self.env.render()

                total_return += episode_return

            fvals[i, 0] = total_return / self.num_rollouts  # negate if you MINIMISE
        return None, fvals


class HopperPolicySearchProblem(BenchmarkProblem):
    available_dimensions = 102  # <-- matches D above
    input_type = DataType.CONTINUOUS
    num_objectives = 1
    num_constraints = 0

    def __init__(
        self,
        num_rollouts: int = 5,
        render: bool = False,
        template_file: Optional[str] = None,
    ):
        # ---------------------------------------------------------------------
        self.env = gym.make("Hopper-v5")
        self.num_rollouts = num_rollouts
        self.render = render
        script_dir = Path(__file__).parent
        template_file = (
            script_dir / "mujoco_policies" / "Hopper-v1" / "lin_policy_plus.npz"
        )

        # -------- template policy, mean, std ---------------------------------
        if template_file is not None:
            arr_0 = np.load(template_file, allow_pickle=True)["arr_0"]
            self.W_shape = arr_0[0].shape  # (2,8)
            self.obs_mean = arr_0[1]
            self.obs_std = arr_0[2]
        else:
            self.W_shape = (
                self.env.action_space.shape[0],
                self.env.observation_space.shape[0],
            )
            self.obs_mean = np.zeros(self.W_shape[1])
            self.obs_std = np.ones(self.W_shape[1])

        dim = int(np.prod(self.W_shape))  # 16

        # ---- element-wise bounds on the weights (same as LA-MCTS code) ------
        lb, ub = -1.0, 1.0  # for Swimmer in MujucoPolicyFunc
        bounds = [(lb, ub)] * dim

        super().__init__(
            dim=dim,
            num_objectives=self.__class__.num_objectives,
            num_constraints=0,
            bounds=bounds,
        )

    # -------------------------------------------------------------------------
    def _evaluate_implementation(
        self, x: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        batch_size = x.shape[0]
        fvals = torch.empty(batch_size, 1, device=x.device)

        for i in range(batch_size):
            # reshape flat vector back to a (2,8) weight matrix
            W = x[i].detach().cpu().numpy().reshape(self.W_shape)

            # average return over N roll-outs (like the Facebook code)
            total_return = 0.0
            for _ in range(self.num_rollouts):
                obs, _ = self.env.reset()
                done = truncated = False
                episode_return = 0.0

                while not (done or truncated):
                    # linear state-feedback control
                    action = np.dot(W, (obs - self.obs_mean) / self.obs_std)
                    # keep the action in the legal torque range
                    action = np.clip(
                        action, self.env.action_space.low, self.env.action_space.high
                    )

                    obs, reward, done, truncated, _ = self.env.step(action)
                    episode_return += reward
                    if self.render:
                        self.env.render()

                total_return += episode_return

            fvals[i, 0] = total_return / self.num_rollouts  # negate if you MINIMISE
        return None, fvals


class Walker2DPolicySearchProblem(BenchmarkProblem):
    available_dimensions = 102  # <-- matches D above
    input_type = DataType.CONTINUOUS
    num_objectives = 1
    num_constraints = 0

    def __init__(
        self,
        num_rollouts: int = 5,
        render: bool = False,
        template_file: Optional[str] = None,
    ):
        # ---------------------------------------------------------------------
        self.env = gym.make("Walker2d-v5")
        self.num_rollouts = num_rollouts
        self.render = render
        script_dir = Path(__file__).parent
        template_file = (
            script_dir / "mujoco_policies" / "Walker2d-v1" / "lin_policy_plus.npz"
        )

        # -------- template policy, mean, std ---------------------------------
        if template_file is not None:
            arr_0 = np.load(template_file, allow_pickle=True)["arr_0"]
            self.W_shape = arr_0[0].shape  # (2,8)
            self.obs_mean = arr_0[1]
            self.obs_std = arr_0[2]
        else:
            self.W_shape = (
                self.env.action_space.shape[0],
                self.env.observation_space.shape[0],
            )
            self.obs_mean = np.zeros(self.W_shape[1])
            self.obs_std = np.ones(self.W_shape[1])

        dim = int(np.prod(self.W_shape))  # 16

        # ---- element-wise bounds on the weights (same as LA-MCTS code) ------
        lb, ub = -1.0, 1.0  # for Swimmer in MujucoPolicyFunc
        bounds = [(lb, ub)] * dim

        super().__init__(
            dim=dim,
            num_objectives=self.__class__.num_objectives,
            num_constraints=0,
            bounds=bounds,
        )

    # -------------------------------------------------------------------------
    def _evaluate_implementation(
        self, x: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        batch_size = x.shape[0]
        fvals = torch.empty(batch_size, 1, device=x.device)

        for i in range(batch_size):
            # reshape flat vector back to a (2,8) weight matrix
            W = x[i].detach().cpu().numpy().reshape(self.W_shape)

            # average return over N roll-outs (like the Facebook code)
            total_return = 0.0
            for _ in range(self.num_rollouts):
                obs, _ = self.env.reset()
                done = truncated = False
                episode_return = 0.0

                while not (done or truncated):
                    # linear state-feedback control
                    action = np.dot(W, (obs - self.obs_mean) / self.obs_std)
                    # keep the action in the legal torque range
                    action = np.clip(
                        action, self.env.action_space.low, self.env.action_space.high
                    )

                    obs, reward, done, truncated, _ = self.env.step(action)
                    episode_return += reward
                    if self.render:
                        self.env.render()

                total_return += episode_return

            fvals[i, 0] = total_return / self.num_rollouts  # negate if you MINIMISE
        return None, fvals
