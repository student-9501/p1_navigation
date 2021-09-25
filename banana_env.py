"""
wrapper to give a unity environment the same interface as aigym to
facilitate running the same agents
"""
import numpy as np
import gym
from unityagents import UnityEnvironment


class Environment:
    action_space = gym.spaces.Discrete(4)
    observation_space = gym.spaces.Box(-np.inf, np.inf, (37,))

    def __init__(self, env_id=0, filename='Banana_Linux_NoVis/Banana.x86_64'):
        self.env = UnityEnvironment(file_name=filename, no_graphics=True, worker_id=env_id)
        self.brain_name = self.env.brain_names[0]

    def reset(self):
        env_info = self.env.reset(train_mode=True)[self.brain_name]
        return env_info.vector_observations[0]

    def step(self, action):
        env_info = self.env.step(action)[self.brain_name]
        next_state = env_info.vector_observations[0]
        reward = env_info.rewards[0]
        done = env_info.local_done[0]
        return next_state, reward, done, None
