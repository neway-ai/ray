import logging
import numpy as np
import torch

from ray.rllib.utils.annotations import override
from ray.rllib.models.catalog import ModelCatalog
from ray.rllib.models.preprocessors import get_preprocessor
from ray.rllib.utils.replay_buffers.multi_agent_replay_buffer import (
    MultiAgentReplayBuffer,
    ReplayMode,
    merge_dicts_with_warning,
)
from ray.rllib.utils.replay_buffers.experimental_replay_buffer import (
    ExperimentalReplayBuffer,
)
from ray.rllib.utils.replay_buffers.replay_buffer import (
    StorageUnit,
)
from ray.rllib.utils.typing import PolicyID, SampleBatchType
from ray.rllib.policy.sample_batch import SampleBatch
from ray.util.debug import log_once
from ray.util.annotations import DeveloperAPI
from ray.rllib.policy.rnn_sequencing import timeslice_along_seq_lens_with_overlap

logger = logging.getLogger(__name__)


@DeveloperAPI
class MultiAgentExperimentalReplayBuffer(
    MultiAgentReplayBuffer, ExperimentalReplayBuffer
):
    def __init__(
        self,
        capacity: int = 10000,
        storage_unit: str = "timesteps",
        num_shards: int = 1,
        replay_mode: str = "independent",
        replay_sequence_override: bool = True,
        replay_sequence_length: int = 1,
        replay_burn_in: int = 0,
        replay_zero_init_states: bool = True,
        underlying_buffer_config: dict = None,
        **kwargs
    ):
        """Initializes a MultiAgentReplayBuffer instance.

        Args:
            capacity: The capacity of the buffer, measured in `storage_unit`.
            storage_unit: Either 'timesteps', 'sequences' or
                'episodes'. Specifies how experiences are stored. If they
                are stored in episodes, replay_sequence_length is ignored.
                If they are stored in episodes, replay_sequence_length is
                ignored.
            num_shards: The number of buffer shards that exist in total
                (including this one).
            replay_mode: One of "independent" or "lockstep". Determines,
                whether batches are sampled independently or to an equal
                amount.
            replay_sequence_override: If True, ignore sequences found in incoming
                batches, slicing them into sequences as specified by
                `replay_sequence_length` and `replay_sequence_burn_in`. This only has
                an effect if storage_unit is `sequences`.
            replay_sequence_length: The sequence length (T) of a single
                sample. If > 1, we will sample B x T from this buffer.
            replay_burn_in: The burn-in length in case
                `replay_sequence_length` > 0. This is the number of timesteps
                each sequence overlaps with the previous one to generate a
                better internal state (=state after the burn-in), instead of
                starting from 0.0 each RNN rollout.
            replay_zero_init_states: Whether the initial states in the
                buffer (if replay_sequence_length > 0) are alwayas 0.0 or
                should be updated with the previous train_batch state outputs.
            underlying_buffer_config: A config that contains all necessary
                constructor arguments and arguments for methods to call on
                the underlying buffers. This replaces the standard behaviour
                of the underlying PrioritizedReplayBuffer. The config
                follows the conventions of the general
                replay_buffer_config. kwargs for subsequent calls of methods
                may also be included. Example:
                "replay_buffer_config": {"type": PrioritizedReplayBuffer,
                "capacity": 10, "storage_unit": "timesteps",
                prioritized_replay_alpha: 0.5, prioritized_replay_beta: 0.5,
                prioritized_replay_eps: 0.5}
            ``**kwargs``: Forward compatibility kwargs.
        """
        if underlying_buffer_config is not None:
            if log_once("underlying_buffer_config_not_supported"):
                logger.info(
                    "PrioritizedMultiAgentReplayBuffer instantiated "
                    "with underlying_buffer_config. This will "
                    "overwrite the standard behaviour of the "
                    "underlying PrioritizedReplayBuffer."
                )
        else:
            underlying_buffer_config = {
                "type": ExperimentalReplayBuffer,
            }

        shard_capacity = capacity // num_shards
        MultiAgentReplayBuffer.__init__(
            self,
            capacity=shard_capacity,
            storage_unit=storage_unit,
            replay_sequence_override=replay_sequence_override,
            replay_mode=replay_mode,
            replay_sequence_length=replay_sequence_length,
            replay_burn_in=replay_burn_in,
            replay_zero_init_states=replay_zero_init_states,
            underlying_buffer_config=underlying_buffer_config,
            **kwargs,
        )
        self.embed_dim = 128
        self.framework = kwargs.get("framework", None)
        self.distill_net_config = kwargs.get("model_config", None)
        env = kwargs.get("env", None)
        import gym

        env = gym.make(env)
        self.action_space = env.action_space
        from ray.rllib.env.wrappers.atari_wrappers import is_atari, wrap_deepmind

        if is_atari(env):
            env = wrap_deepmind(
                env,
                dim=self.distill_net_config["dim"],
                framestack=self.distill_net_config.get("framestack"),
            )
            self.obs_space = env.observation_space
        else:
            self.pp = get_preprocessor(space=env.observation_space)(
                env.observation_space
            )
            self.obs_space = self.pp.observation_space

        self._distill_net = ModelCatalog.get_model_v2(
            self.obs_space,
            self.action_space,
            self.embed_dim,
            model_config=self.distill_net_config,
            framework="torch",
            name="_noveld_distill_net",
        )
        self._distill_target_net = ModelCatalog.get_model_v2(
            self.obs_space,
            self.action_space,
            self.embed_dim,
            model_config=self.distill_net_config,
            framework="torch",
            name="_noveld_distill_target_net",
        )

        # We do not train the target network.
        distill_params = list(self._distill_net.parameters())
        # self.model._noveld_distill_net = self._distill_net.to(self.device)
        self._optimizer = torch.optim.Adam(
            distill_params,
            lr=0.0005,
        )

    @DeveloperAPI
    @override(MultiAgentReplayBuffer)
    def _add_to_underlying_buffer(
        self, policy_id: PolicyID, batch: SampleBatchType, **kwargs
    ) -> None:
        """Add a batch of experiences to the underlying buffer of a policy.

        If the storage unit is `timesteps`, cut the batch into timeslices
        before adding them to the appropriate buffer. Otherwise, let the
        underlying buffer decide how slice batches.

        Args:
            policy_id: ID of the policy that corresponds to the underlying
            buffer
            batch: SampleBatch to add to the underlying buffer
            ``**kwargs``: Forward compatibility kwargs.
        """
        # Merge kwargs, overwriting standard call arguments
        kwargs = merge_dicts_with_warning(self.underlying_buffer_call_args, kwargs)

        if hasattr(self, "pp"):
            transformed = self.pp.transform(batch[SampleBatch.OBS])
        else:
            transformed = batch[SampleBatch.OBS]

        preprocessed_batch = {
            SampleBatch.OBS: torch.from_numpy(transformed),
        }

        phi, _ = self._distill_net(preprocessed_batch)
        phi_target, _ = self._distill_target_net(preprocessed_batch)

        distill_rank = torch.norm(phi - phi_target + 1e-12, dim=1)
        self._distill_rank_np = distill_rank.detach().cpu().numpy()

        # Perform an optimizer step.
        distill_loss = torch.mean(distill_rank)
        self._optimizer.zero_grad()
        distill_loss.backward()
        self._optimizer.step()

        batch["distill_ranks"] = self._distill_rank_np

        # For the storage unit `timesteps`, the underlying buffer will
        # simply store the samples how they arrive. For sequences and
        # episodes, the underlying buffer may split them itself.
        if self.storage_unit is StorageUnit.TIMESTEPS:
            timeslices = batch.timeslices(1)
        elif self.storage_unit is StorageUnit.SEQUENCES:
            timeslices = timeslice_along_seq_lens_with_overlap(
                sample_batch=batch,
                seq_lens=batch.get(SampleBatch.SEQ_LENS)
                if self.replay_sequence_override
                else None,
                zero_pad_max_seq_len=self.replay_sequence_length,
                pre_overlap=self.replay_burn_in,
                zero_init_states=self.replay_zero_init_states,
            )
        elif self.storage_unit == StorageUnit.EPISODES:
            timeslices = []
            for eps in batch.split_by_episode():
                if (
                    eps.get(SampleBatch.T)[0] == 0
                    and eps.get(SampleBatch.DONES)[-1] == True  # noqa E712
                ):
                    # Only add full episodes to the buffer
                    timeslices.append(eps)
                else:
                    if log_once("only_full_episodes"):
                        logger.info(
                            "This buffer uses episodes as a storage "
                            "unit and thus allows only full episodes "
                            "to be added to it. Some samples may be "
                            "dropped."
                        )
        elif self.storage_unit == StorageUnit.FRAGMENTS:
            timeslices = [batch]
        else:
            raise ValueError("Unknown `storage_unit={}`".format(self.storage_unit))

        for slice in timeslices:
            # If SampleBatch has prio-replay weights, average
            # over these to use as a weight for the entire
            # sequence.
            if self.replay_mode is ReplayMode.INDEPENDENT:
                if "weights" in slice and len(slice["weights"]):
                    weight = np.mean(slice["weights"])
                else:
                    weight = None

                kwargs = {"weight": weight, **kwargs}
            else:
                kwargs = {**kwargs, "weight": None}
            self.replay_buffers[policy_id].add(slice, **kwargs)
