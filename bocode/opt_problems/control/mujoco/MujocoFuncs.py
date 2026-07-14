from pathlib import Path

import numpy as np
import torch

try:
    import gymnasium as gym
except ImportError as _exc:  # pragma: no cover - exercised only without the extra
    raise ImportError(
        "MuJoCo control problems require the optional 'mujoco' dependency. "
        "Install it with: pip install 'bocode[mujoco]'"
    ) from _exc

from ....base import BenchmarkProblem

# Constant-action rollout settings, shared by every MuJoCo *Problem variant.
#
# The episode is rolled out from a FIXED reset seed so the benchmark is DETERMINISTIC: MuJoCo's
# reset applies a small random perturbation, and without a fixed seed the same action would score
# differently on every call, which turns the objective into a noisy one and makes the BO
# comparison meaningless.
#
# _MAX_STEPS caps the rollout so a good action cannot run forever (Gymnasium's own time limits are
# 1000 steps for these environments).
_ROLLOUT_SEED = 0
_MAX_STEPS = 1000


class AntProblem(BenchmarkProblem):
    available_dimensions = 8
    num_objectives = 1
    num_constraints = 0
    _rollout_seed = _ROLLOUT_SEED
    _max_steps = _MAX_STEPS

    def __init__(self):
        super().__init__(
            dim=self.__class__.available_dimensions,
            num_objectives=self.__class__.num_objectives,
            num_constraints=0,
            bounds=list(
                zip(
                    gym.make("Ant-v5").action_space.low.tolist(),
                    gym.make("Ant-v5").action_space.high.tolist(),
                    strict=False,
                )
            ),
        )
        self.env = gym.make("Ant-v5")

    def _evaluate_implementation(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        batch_size = x.shape[0]
        rewards = torch.zeros(batch_size, self.__class__.num_objectives)
        for i in range(batch_size):
            action = x[i].cpu().numpy()
            # Roll the EPISODE out under this constant action and return the episode
            # return. Previously this took a SINGLE env.step(), which is not a control
            # task at all -- it scored one timestep from the reset state.
            #
            # It made InvertedPendulum literally constant: its reward is +1 per timestep
            # alive, so one step always returns exactly 1.0. The other ten were not caught
            # only because their per-step rewards happen to be state-dependent -- they were
            # measuring the wrong thing just as badly.
            #   InvertedPendulum, action swept lo->hi:
            #     1 step  : [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]   spread 0.0
            #     episode : [2.0, 2.0, 3.0, 23.0, 3.0, 2.0, 2.0]  spread 21.0  (optimum at a=0)
            #
            # Gymnasium MuJoCo returns are to be MAXIMIZED, which is already BoCoDe's
            # convention -- return the episode return as-is (do not negate).
            self.env.reset(seed=self._rollout_seed)
            done = truncated = False
            episode_return = 0.0
            steps = 0
            while not (done or truncated) and steps < self._max_steps:
                _, reward, done, truncated, _ = self.env.step(action)
                episode_return += float(reward)
                steps += 1
            rewards[i, 0] = episode_return
        return None, rewards


class HalfCheetahProblem(BenchmarkProblem):
    available_dimensions = 6
    num_objectives = 1
    num_constraints = 0
    _rollout_seed = _ROLLOUT_SEED
    _max_steps = _MAX_STEPS

    def __init__(self):
        super().__init__(
            dim=self.__class__.available_dimensions,
            num_objectives=self.__class__.num_objectives,
            num_constraints=0,
            bounds=list(
                zip(
                    gym.make("HalfCheetah-v5").action_space.low.tolist(),
                    gym.make("HalfCheetah-v5").action_space.high.tolist(),
                    strict=False,
                )
            ),
        )
        self.env = gym.make("HalfCheetah-v5")

    def _evaluate_implementation(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        batch_size = x.shape[0]
        rewards = torch.zeros(batch_size, self.__class__.num_objectives)
        for i in range(batch_size):
            action = x[i].cpu().numpy()
            # Roll the EPISODE out under this constant action and return the episode
            # return. Previously this took a SINGLE env.step(), which is not a control
            # task at all -- it scored one timestep from the reset state.
            #
            # It made InvertedPendulum literally constant: its reward is +1 per timestep
            # alive, so one step always returns exactly 1.0. The other ten were not caught
            # only because their per-step rewards happen to be state-dependent -- they were
            # measuring the wrong thing just as badly.
            #   InvertedPendulum, action swept lo->hi:
            #     1 step  : [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]   spread 0.0
            #     episode : [2.0, 2.0, 3.0, 23.0, 3.0, 2.0, 2.0]  spread 21.0  (optimum at a=0)
            #
            # Gymnasium MuJoCo returns are to be MAXIMIZED, which is already BoCoDe's
            # convention -- return the episode return as-is (do not negate).
            self.env.reset(seed=self._rollout_seed)
            done = truncated = False
            episode_return = 0.0
            steps = 0
            while not (done or truncated) and steps < self._max_steps:
                _, reward, done, truncated, _ = self.env.step(action)
                episode_return += float(reward)
                steps += 1
            rewards[i, 0] = episode_return
        return None, rewards


class HopperProblem(BenchmarkProblem):
    available_dimensions = 3
    num_objectives = 1
    num_constraints = 0
    _rollout_seed = _ROLLOUT_SEED
    _max_steps = _MAX_STEPS

    def __init__(self):
        super().__init__(
            dim=self.__class__.available_dimensions,
            num_objectives=self.__class__.num_objectives,
            num_constraints=0,
            bounds=list(
                zip(
                    gym.make("Hopper-v5").action_space.low.tolist(),
                    gym.make("Hopper-v5").action_space.high.tolist(),
                    strict=False,
                )
            ),
        )
        self.env = gym.make("Hopper-v5")

    def _evaluate_implementation(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        batch_size = x.shape[0]
        rewards = torch.zeros(batch_size, self.__class__.num_objectives)
        for i in range(batch_size):
            action = x[i].cpu().numpy()
            # Roll the EPISODE out under this constant action and return the episode
            # return. Previously this took a SINGLE env.step(), which is not a control
            # task at all -- it scored one timestep from the reset state.
            #
            # It made InvertedPendulum literally constant: its reward is +1 per timestep
            # alive, so one step always returns exactly 1.0. The other ten were not caught
            # only because their per-step rewards happen to be state-dependent -- they were
            # measuring the wrong thing just as badly.
            #   InvertedPendulum, action swept lo->hi:
            #     1 step  : [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]   spread 0.0
            #     episode : [2.0, 2.0, 3.0, 23.0, 3.0, 2.0, 2.0]  spread 21.0  (optimum at a=0)
            #
            # Gymnasium MuJoCo returns are to be MAXIMIZED, which is already BoCoDe's
            # convention -- return the episode return as-is (do not negate).
            self.env.reset(seed=self._rollout_seed)
            done = truncated = False
            episode_return = 0.0
            steps = 0
            while not (done or truncated) and steps < self._max_steps:
                _, reward, done, truncated, _ = self.env.step(action)
                episode_return += float(reward)
                steps += 1
            rewards[i, 0] = episode_return
        return None, rewards


class HumanoidProblem(BenchmarkProblem):
    available_dimensions = 17
    num_objectives = 1
    num_constraints = 0
    _rollout_seed = _ROLLOUT_SEED
    _max_steps = _MAX_STEPS

    def __init__(self):
        super().__init__(
            dim=self.__class__.available_dimensions,
            num_objectives=self.__class__.num_objectives,
            num_constraints=0,
            bounds=list(
                zip(
                    gym.make("Humanoid-v5").action_space.low.tolist(),
                    gym.make("Humanoid-v5").action_space.high.tolist(),
                    strict=False,
                )
            ),
        )
        self.env = gym.make("Humanoid-v5")

    def _evaluate_implementation(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        batch_size = x.shape[0]
        rewards = torch.zeros(batch_size, self.__class__.num_objectives)
        for i in range(batch_size):
            action = x[i].cpu().numpy()
            # Roll the EPISODE out under this constant action and return the episode
            # return. Previously this took a SINGLE env.step(), which is not a control
            # task at all -- it scored one timestep from the reset state.
            #
            # It made InvertedPendulum literally constant: its reward is +1 per timestep
            # alive, so one step always returns exactly 1.0. The other ten were not caught
            # only because their per-step rewards happen to be state-dependent -- they were
            # measuring the wrong thing just as badly.
            #   InvertedPendulum, action swept lo->hi:
            #     1 step  : [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]   spread 0.0
            #     episode : [2.0, 2.0, 3.0, 23.0, 3.0, 2.0, 2.0]  spread 21.0  (optimum at a=0)
            #
            # Gymnasium MuJoCo returns are to be MAXIMIZED, which is already BoCoDe's
            # convention -- return the episode return as-is (do not negate).
            self.env.reset(seed=self._rollout_seed)
            done = truncated = False
            episode_return = 0.0
            steps = 0
            while not (done or truncated) and steps < self._max_steps:
                _, reward, done, truncated, _ = self.env.step(action)
                episode_return += float(reward)
                steps += 1
            rewards[i, 0] = episode_return
        return None, rewards


class HumanoidStandupProblem(BenchmarkProblem):
    available_dimensions = 17
    num_objectives = 1
    num_constraints = 0
    _rollout_seed = _ROLLOUT_SEED
    _max_steps = _MAX_STEPS

    def __init__(self):
        super().__init__(
            dim=self.__class__.available_dimensions,
            num_objectives=self.__class__.num_objectives,
            num_constraints=0,
            bounds=list(
                zip(
                    gym.make("HumanoidStandup-v5").action_space.low.tolist(),
                    gym.make("HumanoidStandup-v5").action_space.high.tolist(),
                    strict=False,
                )
            ),
        )
        self.env = gym.make("HumanoidStandup-v5")

    def _evaluate_implementation(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        batch_size = x.shape[0]
        rewards = torch.zeros(batch_size, self.__class__.num_objectives)
        for i in range(batch_size):
            action = x[i].cpu().numpy()
            # Roll the EPISODE out under this constant action and return the episode
            # return. Previously this took a SINGLE env.step(), which is not a control
            # task at all -- it scored one timestep from the reset state.
            #
            # It made InvertedPendulum literally constant: its reward is +1 per timestep
            # alive, so one step always returns exactly 1.0. The other ten were not caught
            # only because their per-step rewards happen to be state-dependent -- they were
            # measuring the wrong thing just as badly.
            #   InvertedPendulum, action swept lo->hi:
            #     1 step  : [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]   spread 0.0
            #     episode : [2.0, 2.0, 3.0, 23.0, 3.0, 2.0, 2.0]  spread 21.0  (optimum at a=0)
            #
            # Gymnasium MuJoCo returns are to be MAXIMIZED, which is already BoCoDe's
            # convention -- return the episode return as-is (do not negate).
            self.env.reset(seed=self._rollout_seed)
            done = truncated = False
            episode_return = 0.0
            steps = 0
            while not (done or truncated) and steps < self._max_steps:
                _, reward, done, truncated, _ = self.env.step(action)
                episode_return += float(reward)
                steps += 1
            rewards[i, 0] = episode_return
        return None, rewards


class InvertedDoublePendulumProblem(BenchmarkProblem):
    available_dimensions = 1
    num_objectives = 1
    num_constraints = 0
    _rollout_seed = _ROLLOUT_SEED
    _max_steps = _MAX_STEPS

    def __init__(self):
        super().__init__(
            dim=self.__class__.available_dimensions,
            num_objectives=self.__class__.num_objectives,
            num_constraints=0,
            bounds=list(
                zip(
                    gym.make("InvertedDoublePendulum-v5").action_space.low.tolist(),
                    gym.make("InvertedDoublePendulum-v5").action_space.high.tolist(),
                    strict=False,
                )
            ),
        )
        self.env = gym.make("InvertedDoublePendulum-v5")

    def _evaluate_implementation(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        batch_size = x.shape[0]
        rewards = torch.zeros(batch_size, self.__class__.num_objectives)
        for i in range(batch_size):
            action = x[i].cpu().numpy()
            # Roll the EPISODE out under this constant action and return the episode
            # return. Previously this took a SINGLE env.step(), which is not a control
            # task at all -- it scored one timestep from the reset state.
            #
            # It made InvertedPendulum literally constant: its reward is +1 per timestep
            # alive, so one step always returns exactly 1.0. The other ten were not caught
            # only because their per-step rewards happen to be state-dependent -- they were
            # measuring the wrong thing just as badly.
            #   InvertedPendulum, action swept lo->hi:
            #     1 step  : [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]   spread 0.0
            #     episode : [2.0, 2.0, 3.0, 23.0, 3.0, 2.0, 2.0]  spread 21.0  (optimum at a=0)
            #
            # Gymnasium MuJoCo returns are to be MAXIMIZED, which is already BoCoDe's
            # convention -- return the episode return as-is (do not negate).
            self.env.reset(seed=self._rollout_seed)
            done = truncated = False
            episode_return = 0.0
            steps = 0
            while not (done or truncated) and steps < self._max_steps:
                _, reward, done, truncated, _ = self.env.step(action)
                episode_return += float(reward)
                steps += 1
            rewards[i, 0] = episode_return
        return None, rewards


class InvertedPendulumProblem(BenchmarkProblem):
    available_dimensions = 1
    num_objectives = 1
    num_constraints = 0
    _rollout_seed = _ROLLOUT_SEED
    _max_steps = _MAX_STEPS

    def __init__(self):
        super().__init__(
            dim=self.__class__.available_dimensions,
            num_objectives=self.__class__.num_objectives,
            num_constraints=0,
            bounds=list(
                zip(
                    gym.make("InvertedPendulum-v5").action_space.low.tolist(),
                    gym.make("InvertedPendulum-v5").action_space.high.tolist(),
                    strict=False,
                )
            ),
        )
        self.env = gym.make("InvertedPendulum-v5")

    def _evaluate_implementation(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        batch_size = x.shape[0]
        rewards = torch.zeros(batch_size, self.__class__.num_objectives)
        for i in range(batch_size):
            action = x[i].cpu().numpy()
            # Roll the EPISODE out under this constant action and return the episode
            # return. Previously this took a SINGLE env.step(), which is not a control
            # task at all -- it scored one timestep from the reset state.
            #
            # It made InvertedPendulum literally constant: its reward is +1 per timestep
            # alive, so one step always returns exactly 1.0. The other ten were not caught
            # only because their per-step rewards happen to be state-dependent -- they were
            # measuring the wrong thing just as badly.
            #   InvertedPendulum, action swept lo->hi:
            #     1 step  : [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]   spread 0.0
            #     episode : [2.0, 2.0, 3.0, 23.0, 3.0, 2.0, 2.0]  spread 21.0  (optimum at a=0)
            #
            # Gymnasium MuJoCo returns are to be MAXIMIZED, which is already BoCoDe's
            # convention -- return the episode return as-is (do not negate).
            self.env.reset(seed=self._rollout_seed)
            done = truncated = False
            episode_return = 0.0
            steps = 0
            while not (done or truncated) and steps < self._max_steps:
                _, reward, done, truncated, _ = self.env.step(action)
                episode_return += float(reward)
                steps += 1
            rewards[i, 0] = episode_return
        return None, rewards


class PusherProblem(BenchmarkProblem):
    available_dimensions = 7
    num_objectives = 1
    num_constraints = 0
    _rollout_seed = _ROLLOUT_SEED
    _max_steps = _MAX_STEPS

    def __init__(self):
        super().__init__(
            dim=self.__class__.available_dimensions,
            num_objectives=self.__class__.num_objectives,
            num_constraints=0,
            bounds=list(
                zip(
                    gym.make("Pusher-v5").action_space.low.tolist(),
                    gym.make("Pusher-v5").action_space.high.tolist(),
                    strict=False,
                )
            ),
        )
        self.env = gym.make("Pusher-v5")

    def _evaluate_implementation(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        batch_size = x.shape[0]
        rewards = torch.zeros(batch_size, self.__class__.num_objectives)
        for i in range(batch_size):
            action = x[i].cpu().numpy()
            # Roll the EPISODE out under this constant action and return the episode
            # return. Previously this took a SINGLE env.step(), which is not a control
            # task at all -- it scored one timestep from the reset state.
            #
            # It made InvertedPendulum literally constant: its reward is +1 per timestep
            # alive, so one step always returns exactly 1.0. The other ten were not caught
            # only because their per-step rewards happen to be state-dependent -- they were
            # measuring the wrong thing just as badly.
            #   InvertedPendulum, action swept lo->hi:
            #     1 step  : [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]   spread 0.0
            #     episode : [2.0, 2.0, 3.0, 23.0, 3.0, 2.0, 2.0]  spread 21.0  (optimum at a=0)
            #
            # Gymnasium MuJoCo returns are to be MAXIMIZED, which is already BoCoDe's
            # convention -- return the episode return as-is (do not negate).
            self.env.reset(seed=self._rollout_seed)
            done = truncated = False
            episode_return = 0.0
            steps = 0
            while not (done or truncated) and steps < self._max_steps:
                _, reward, done, truncated, _ = self.env.step(action)
                episode_return += float(reward)
                steps += 1
            rewards[i, 0] = episode_return
        return None, rewards


class ReacherProblem(BenchmarkProblem):
    available_dimensions = 2
    num_objectives = 1
    num_constraints = 0
    _rollout_seed = _ROLLOUT_SEED
    _max_steps = _MAX_STEPS

    def __init__(self):
        super().__init__(
            dim=self.__class__.available_dimensions,
            num_objectives=self.__class__.num_objectives,
            num_constraints=0,
            bounds=list(
                zip(
                    gym.make("Reacher-v5").action_space.low.tolist(),
                    gym.make("Reacher-v5").action_space.high.tolist(),
                    strict=False,
                )
            ),
        )
        self.env = gym.make("Reacher-v5")

    def _evaluate_implementation(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        batch_size = x.shape[0]
        rewards = torch.zeros(batch_size, self.__class__.num_objectives)
        for i in range(batch_size):
            action = x[i].cpu().numpy()
            # Roll the EPISODE out under this constant action and return the episode
            # return. Previously this took a SINGLE env.step(), which is not a control
            # task at all -- it scored one timestep from the reset state.
            #
            # It made InvertedPendulum literally constant: its reward is +1 per timestep
            # alive, so one step always returns exactly 1.0. The other ten were not caught
            # only because their per-step rewards happen to be state-dependent -- they were
            # measuring the wrong thing just as badly.
            #   InvertedPendulum, action swept lo->hi:
            #     1 step  : [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]   spread 0.0
            #     episode : [2.0, 2.0, 3.0, 23.0, 3.0, 2.0, 2.0]  spread 21.0  (optimum at a=0)
            #
            # Gymnasium MuJoCo returns are to be MAXIMIZED, which is already BoCoDe's
            # convention -- return the episode return as-is (do not negate).
            self.env.reset(seed=self._rollout_seed)
            done = truncated = False
            episode_return = 0.0
            steps = 0
            while not (done or truncated) and steps < self._max_steps:
                _, reward, done, truncated, _ = self.env.step(action)
                episode_return += float(reward)
                steps += 1
            rewards[i, 0] = episode_return
        return None, rewards


class Walker2DProblem(BenchmarkProblem):
    available_dimensions = 6
    num_objectives = 1
    num_constraints = 0
    _rollout_seed = _ROLLOUT_SEED
    _max_steps = _MAX_STEPS

    def __init__(self):
        super().__init__(
            dim=self.__class__.available_dimensions,
            num_objectives=self.__class__.num_objectives,
            num_constraints=0,
            bounds=list(
                zip(
                    gym.make("Walker2d-v5").action_space.low.tolist(),
                    gym.make("Walker2d-v5").action_space.high.tolist(),
                    strict=False,
                )
            ),
        )
        self.env = gym.make("Walker2d-v5")

    def _evaluate_implementation(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        batch_size = x.shape[0]
        rewards = torch.zeros(batch_size, self.__class__.num_objectives)
        for i in range(batch_size):
            action = x[i].cpu().numpy()
            # Roll the EPISODE out under this constant action and return the episode
            # return. Previously this took a SINGLE env.step(), which is not a control
            # task at all -- it scored one timestep from the reset state.
            #
            # It made InvertedPendulum literally constant: its reward is +1 per timestep
            # alive, so one step always returns exactly 1.0. The other ten were not caught
            # only because their per-step rewards happen to be state-dependent -- they were
            # measuring the wrong thing just as badly.
            #   InvertedPendulum, action swept lo->hi:
            #     1 step  : [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]   spread 0.0
            #     episode : [2.0, 2.0, 3.0, 23.0, 3.0, 2.0, 2.0]  spread 21.0  (optimum at a=0)
            #
            # Gymnasium MuJoCo returns are to be MAXIMIZED, which is already BoCoDe's
            # convention -- return the episode return as-is (do not negate).
            self.env.reset(seed=self._rollout_seed)
            done = truncated = False
            episode_return = 0.0
            steps = 0
            while not (done or truncated) and steps < self._max_steps:
                _, reward, done, truncated, _ = self.env.step(action)
                episode_return += float(reward)
                steps += 1
            rewards[i, 0] = episode_return
        return None, rewards


class SwimmerProblem(BenchmarkProblem):
    available_dimensions = 2
    num_objectives = 1
    num_constraints = 0
    _rollout_seed = _ROLLOUT_SEED
    _max_steps = _MAX_STEPS

    def __init__(self):
        super().__init__(
            dim=self.__class__.available_dimensions,
            num_objectives=self.__class__.num_objectives,
            num_constraints=0,
            bounds=list(
                zip(
                    gym.make("Swimmer-v5").action_space.low.tolist(),
                    gym.make("Swimmer-v5").action_space.high.tolist(),
                    strict=False,
                )
            ),
        )
        self.env = gym.make("Swimmer-v5")

    def _evaluate_implementation(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        batch_size = x.shape[0]
        rewards = torch.zeros(batch_size, self.__class__.num_objectives)
        for i in range(batch_size):
            action = x[i].cpu().numpy()
            # Roll the EPISODE out under this constant action and return the episode
            # return. Previously this took a SINGLE env.step(), which is not a control
            # task at all -- it scored one timestep from the reset state.
            #
            # It made InvertedPendulum literally constant: its reward is +1 per timestep
            # alive, so one step always returns exactly 1.0. The other ten were not caught
            # only because their per-step rewards happen to be state-dependent -- they were
            # measuring the wrong thing just as badly.
            #   InvertedPendulum, action swept lo->hi:
            #     1 step  : [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]   spread 0.0
            #     episode : [2.0, 2.0, 3.0, 23.0, 3.0, 2.0, 2.0]  spread 21.0  (optimum at a=0)
            #
            # Gymnasium MuJoCo returns are to be MAXIMIZED, which is already BoCoDe's
            # convention -- return the episode return as-is (do not negate).
            self.env.reset(seed=self._rollout_seed)
            done = truncated = False
            episode_return = 0.0
            steps = 0
            while not (done or truncated) and steps < self._max_steps:
                _, reward, done, truncated, _ = self.env.step(action)
                episode_return += float(reward)
                steps += 1
            rewards[i, 0] = episode_return
        return None, rewards


class SwimmerPolicySearchProblem(BenchmarkProblem):
    available_dimensions = 16  # <-- matches D above
    num_objectives = 1
    num_constraints = 0
    _rollout_seed = _ROLLOUT_SEED
    _max_steps = _MAX_STEPS

    def __init__(
        self,
        num_rollouts: int = 5,
        render: bool = False,
        template_file: str | None = None,
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
    ) -> tuple[torch.Tensor, torch.Tensor]:
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


class AntPolicySearchProblem(BenchmarkProblem):
    available_dimensions = 840  # <-- matches D above
    num_objectives = 1
    num_constraints = 0
    _rollout_seed = _ROLLOUT_SEED
    _max_steps = _MAX_STEPS

    def __init__(
        self,
        num_rollouts: int = 5,
        render: bool = False,
        template_file: str | None = None,
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
    ) -> tuple[torch.Tensor, torch.Tensor]:
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
    num_objectives = 1
    num_constraints = 0
    _rollout_seed = _ROLLOUT_SEED
    _max_steps = _MAX_STEPS

    def __init__(
        self,
        num_rollouts: int = 5,
        render: bool = False,
        template_file: str | None = None,
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
    ) -> tuple[torch.Tensor, torch.Tensor]:
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
    num_objectives = 1
    num_constraints = 0
    _rollout_seed = _ROLLOUT_SEED
    _max_steps = _MAX_STEPS

    def __init__(
        self,
        num_rollouts: int = 5,
        render: bool = False,
        template_file: str | None = None,
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
    ) -> tuple[torch.Tensor, torch.Tensor]:
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
    num_objectives = 1
    num_constraints = 0
    _rollout_seed = _ROLLOUT_SEED
    _max_steps = _MAX_STEPS

    def __init__(
        self,
        num_rollouts: int = 5,
        render: bool = False,
        template_file: str | None = None,
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
    ) -> tuple[torch.Tensor, torch.Tensor]:
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
