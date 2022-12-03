import ray
import ray.rllib.algorithms.dqn as dqn
from ray import air, tune
from ray.rllib.utils.test_utils import check_train_results

use_tune = True
num_iterations = 400

config = dqn.DQNConfig().rollouts(num_rollout_workers=0).framework("torch")
buffer_config = config.replay_buffer_config.update(
    {"type": "MultiAgentReplayBuffer", "capacity": 50000}
)
experimental_buffer_config = config.replay_buffer_config.update(
    {
        "type": "MultiAgentExperimentalReplayBuffer",
        "capacity": 50000,
    }
)
prioritized_buffer_config = config.replay_buffer_config.update(
    {
        "type": "MultiAgentPrioritizedReplayBuffer",
        "capacity": 50000,
    }
)
config.environment(env="LunarLander-v2")
config.debugging(seed=tune.grid_search([42, 348]))

if not use_tune:

    config.training(replay_buffer_config=buffer_config)
    trainer = dqn.DQN(config=config)

    ray.init(local_mode=True, ignore_reinit_error=True)
    for i in range(num_iterations):
        results = trainer.train()
        check_train_results(results)
        print(results)

    trainer.stop()
    ray.shutdown()
else:
    # Use either Experimental- or Prioritized buffer.
    buffer_config = config.replay_buffer_config.update(
        {
            "type": tune.grid_search(
                [
                    "MultiAgentExperimentalReplayBuffer",
                    "MultiAgentPrioritizedReplayBuffer",
                    "MultiAgentReplayBuffer",
                ]
            )
        }
    )
    config.training(replay_buffer_config=buffer_config)
    tuner = tune.Tuner(
        "DQN",
        run_config=air.RunConfig(
            stop={"training_iteration": num_iterations},
        ),
        param_space=config.to_dict(),
    )
    results = tuner.fit()
