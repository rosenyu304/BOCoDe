import torch
import numpy as np
import gym

from ..base import BenchmarkProblem, DataType


class PD4CartPole(BenchmarkProblem):
    """
    Source: https://github.com/PREDICT-EPFL/ECC24_BOmeetsControl_workshop/blob/master/exercise2.ipynb
    By Wenjie Xu
    """

    available_dimensions = 4
    input_type = DataType.CONTINUOUS
    num_objectives = 1
    num_constraints = 0

    # 4D objective, 0 constraints, X = n-by-4

    tags = {"single_objective", "unconstrained", "4D"}

    def __init__(self):
        super().__init__(
            dim=4,
            num_objectives=1,
            num_constraints=0,
            bounds=[
                (-5.0, 5.0),  # bounds for kp
                (-3.0, 3.0),  # bounds for kd
                (-5.0, 5.0),  # bounds for kpx
                (-3.0, 3.0),  # bounds for kdx
            ],
        )

    def _evaluate_implementation(self, X, scaling=False):
        if scaling:
            X = super().scale(X)

        n = X.size(0)
        rewards = torch.zeros(n, dtype=torch.float)

        # This loop evaluates each parameter set (each row in X)
        for i in range(n):
            Kp, Kd, Kpx, Kdx = (
                X[i, 0].item(),
                X[i, 1].item(),
                X[i, 2].item(),
                X[i, 3].item(),
            )

            env = gym.make("CartPole-v1")

            def pd_controller(state):
                # Extract the state variables
                x, x_dot, theta, theta_dot = state

                # PD control law
                force = Kp * theta + Kd * theta_dot + Kpx * x + Kdx * x_dot

                # Convert force to discrete action (left or right)
                action = 1 if force > 0 else 0

                return action

            num_episodes = 5
            max_steps = 500
            total_reward_list = []
            for episode in range(num_episodes):
                state = env.reset()
                state = state[0]
                total_reward = 0
                for t in range(max_steps):
                    # Get action from PD controller
                    action = pd_controller(state)

                    # Step the environment
                    state, reward, done, _, _ = env.step(action)

                    total_reward += max(
                        (reward - 70 * np.abs(state[2]) - 5 * np.abs(state[1])), 0
                    )  # penalize the pole angle and moving

                    if done:
                        break

                # print(f"Episode {episode + 1}: Total Reward = {total_reward}")
                total_reward_list.append(total_reward)
            env.close()
            rewards[i] = np.min(total_reward_list)

        fx = rewards.reshape(n, 1)
        return None, fx
