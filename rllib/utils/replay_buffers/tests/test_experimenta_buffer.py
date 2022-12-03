import gym
import unittest

from ray import tune, air
from ray.rllib.algorithms.dqn import DQN, DQNConfig


class TestExperimentalReplayBuffer(unittest.TestCase):
    def test_execution(self):
        dummy_env = gym.make("CartPole-v0")
        action_space = dummy_env.action_space
        observation_space = dummy_env.observation_space

        config = DQNConfig().environment(env="CartPole-v0",
                                         observation_space=observation_space,
                                         action_space=action_space)

        buffer_config = {
                "type": "MultiAgentExperimentalReplayBuffer",
                "observation_space": observation_space,
                "action_space": action_space,
                "model_config": config.model,
        }
        stop_config = {
            "training_iteration": 10,
        }

        config.training(replay_buffer_config=buffer_config)

        tuner = tune.Tuner(
            DQN,
            param_space=config,
            run_config=air.RunConfig(
                stop=stop_config,
            ),
        )

        tuner.fit()
