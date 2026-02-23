import gymnasium as gym
print([env_id for env_id in gym.envs.registry.keys() if 'dm_control' in env_id])