import gymnasium as gym
from functools import wraps
from collections.abc import Callable

'''
Rollout Boundary Truncation (RBT) Wrapper
Stops the running episode and resets the environment
when maximum allowed experience has been aqcuired.

rollout_steps: Maximum allowed rollouts (experience)
total_step_count: Total step counter during experience acquisition 
'''
class RbtWrapper(gym.Wrapper):
    def __init__(self, env: gym.Env, rollout_steps: int):
        super().__init__(env)
        self.rollout_steps = rollout_steps
        self.total_step_count = 0

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        self.total_step_count += 1

        if self.total_step_count == self.rollout_steps:
            self.total_step_count = 0

            if not terminated and not truncated:
                truncated = True
                info = {**info, "rollout_boundary": True}

        return obs, reward, terminated, truncated, info


def with_rollout_boundary(fun: Callable[[], gym.Env]):
    @wraps(fun)
    def create_env_with_rbt(rollout_steps: int = 600):
        env = fun()
        return RbtWrapper(env, rollout_steps=rollout_steps)
    return create_env_with_rbt