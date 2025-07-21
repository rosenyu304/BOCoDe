import torch
import numpy as np
import gym

from ..base import BenchmarkProblem, DataType


class PID4Acrobot(BenchmarkProblem):
    """
    Source: https://github.com/PREDICT-EPFL/ECC24_BOmeetsControl_workshop/blob/master/demo2.ipynb
    By Wenjie Xu

    BoTorch BenchmarkProblem for tuning a PID controller on the
    OpenAI Gym Acrobot-v1 environment.

    The goal is to find the PID gains (kp, ki, kd) that maximize the
    cumulative reward, which corresponds to swinging the acrobot's
    free end to a target height as quickly as possible.

    Note: The evaluation function is stochastic, averaging the outcome
    over 5 distinct simulation seeds. The action space is also handled
    in a non-standard way by casting the continuous PID output to a
    discrete action {0, 1, 2}.
    """

    available_dimensions = 3
    input_type = DataType.CONTINUOUS
    num_objectives = 1
    num_constraints = 0

    # 3D objective, 0 constraints, X = n-by-3

    tags = {"single_objective", "unconstrained", "3D"}

    def __init__(self):
        super().__init__(
            dim=3,
            num_objectives=1,
            num_constraints=0,
            bounds=[
                (-5.0, 5.0),  # bounds for kp
                (-5.0, 5.0),  # bounds for ki
                (-3.0, 3.0),  # bounds for kd
            ],
        )

    def _compute_pid_control(self, kp, ki, kd, target, current, prev_error, integral):
        """Helper function to compute a single PID control step."""
        error = target - current
        integral += error
        derivative = error - prev_error
        control_signal = kp * error + ki * integral + kd * derivative
        return control_signal, error, integral

    def _evaluate_implementation(self, X, scaling=False):
        """
        Evaluates each parameter set in X by running the Acrobot simulation.

        Args:
            X (torch.Tensor): An n-by-3 tensor where each row is a set of
                              [kp, ki, kd] parameters.
            scaling (bool): If True, scales the input X.

        Returns:
            A tuple (gx, fx) where gx is for constraints (empty here) and
            fx is the objective value. Since the original problem maximizes
            reward, this function returns the negative of the mean reward
            to frame it as a minimization problem.
        """
        if scaling:
            # Assumes base class handles scaling from [0,1] to parameter bounds
            X = super().scale(X)

        n = X.size(0)
        rewards = torch.zeros(n, dtype=torch.float)

        # This loop evaluates each parameter set (each row in X)
        for i in range(n):
            kp, ki, kd = X[i, 0].item(), X[i, 1].item(), X[i, 2].item()

            env = gym.make("Acrobot-v1")

            epoch_num = 5  # Number of runs to average over for robustness
            total_reward_list = []
            seed_list = [20 + k for k in range(epoch_num)]

            for epoch_id in range(epoch_num):
                seed = seed_list[epoch_id]
                total_reward = 0

                # The gym environment API has changed slightly over time.
                # env.seed() and env.action_space.seed() are deprecated.
                # The modern approach is to pass the seed to reset().
                # However, to match the notebook's logic, we use the older methods.
                if hasattr(env, "seed"):
                    env.seed(seed)
                    env.action_space.seed(seed)

                env.reset(seed=seed)

                prev_error = 0
                integral = 0

                for _ in range(500):  # Max steps per episode
                    state = env.state

                    # Compute PID control signal for the first joint angle
                    control_signal, prev_error, integral = self._compute_pid_control(
                        kp, ki, kd, 0, state[0], prev_error, integral
                    )

                    # The original notebook clips and casts the action
                    action = max(min(int(control_signal), 2), 0)

                    # Step the environment
                    _next_state, reward, done, _truncated, _info = env.step(action)

                    if done:
                        break
                    total_reward += reward

                total_reward_list.append(total_reward)

            env.close()
            rewards[i] = np.mean(total_reward_list)

        fx = rewards.reshape(n, self.num_objectives)

        return None, fx
