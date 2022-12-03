from ray.rllib.algorithms.dqn import DQN, DQNConfig
import gym

dummy_env = gym.make("CartPole-v0")
action_space = dummy_env.action_space
observation_space = dummy_env.observation_space

config = DQNConfig().environment(
    env="CartPole-v0", observation_space=observation_space, action_space=action_space
)

buffer_config = {
    "type": "MultiAgentExperimentalReplayBuffer",
    "observation_space": observation_space,
    "action_space": action_space,
    "model_config": config.model,
}

config.training(replay_buffer_config=buffer_config)

trainer = DQN("Pendulum-v1", config=config.to_dict())
