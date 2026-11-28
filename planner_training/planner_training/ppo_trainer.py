#!/home/ktingos/.venvs/planner-ppo/bin/python

from planner_gym.adaptive_planner_env import AdaptivePlannerEnv
from extractor_network import NavFeatExtractor
from planner_gym.wrappers import with_rollout_boundary
from stable_baselines3 import PPO
from functools import wraps

import gymnasium as gym

@with_rollout_boundary
def create_env() -> gym.Env:
    return AdaptivePlannerEnv()

def main():
    ROLLOUT_STEPS = 600
    NUM_ROLLOUTS = 10

    env = create_env(rollout_steps=ROLLOUT_STEPS)

    model = PPO(
        policy = 'MlpPolicy',
        env = env,
        n_steps = ROLLOUT_STEPS,
        batch_size=60,
        n_epochs=10,
        policy_kwargs = {
            'feature_class_extractor': NavFeatExtractor,
            'net_arch': []
        },
        verbose=1
    )

    try:
        model.learn(total_timesteps=ROLLOUT_STEPS * NUM_ROLLOUTS)
        model.save("planner_ppo")
    finally:
        env.close()

if __name__ == '__main__':
    main()