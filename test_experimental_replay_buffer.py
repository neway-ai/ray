import ray
import ray.rllib.algorithms.dqn as dqn

from ray.rllib.utils.test_utils import (
    check,
    check_compute_single_action,
    check_train_results,
    framework_iterator,
)

buffer_config = {
    "MultiAgentExperimentalReplayBuffer"
}
config = (
    dqn.DQNConfig()
    .rollouts(num_rollout_workers=0)
    .training(replay_buffer_config=buffer_config)
    .framework("torch")
)

trainer = dqn.DQN(config=config, env="LunarLander-v2")

ray.init(local_mode=True, ignore_reinit_error=True)
for i in range(10):
    results = trainer.train()
    check_train_results(results)
    print(results) 
    
trainer.stop()
ray.shutdown()