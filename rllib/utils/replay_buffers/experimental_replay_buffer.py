import logging
from typing import Union
from sortedcontainers import SortedDict

from ray.rllib.utils.replay_buffers.replay_buffer import ReplayBuffer, StorageUnit, warn_replay_capacity
from ray.rllib.policy.sample_batch import SampleBatch
from ray.util.debug import log_once
from ray.util.annotations import DeveloperAPI
from ray.rllib.utils.typing import SampleBatchType, T

logger = logging.getLogger(__name__)

class ExperimentalReplayBuffer(ReplayBuffer):
    
    def __init__(
        self,
        capacity: int = 10000,
        storage_unit: Union[str, StorageUnit] = "timesteps",
        **kwargs,
    ):
        super().__init__(capacity=capacity, storage_unit=storage_unit, **kwargs)
        
        self._distill_index_map = SortedDict()
        self._hit = False
        
        # Define some metrics.
    
    @DeveloperAPI
    def add(self, batch: SampleBatchType, **kwargs) -> None:
        """Adds a batch of experiences to this buffer.

        Splits batch into chunks of timesteps, sequences or episodes, depending on
        `self._storage_unit`. Calls `self._add_single_batch` to add resulting slices
        to the buffer storage.

        Args:
            batch: Batch to add.
            ``**kwargs``: Forward compatibility kwargs.
        """
        if not batch.count > 0:
            return

        warn_replay_capacity(item=batch, num_items=self.capacity / batch.count)
        
        distill_ranks = batch["distill_ranks"].tolist()
        if self.storage_unit == StorageUnit.TIMESTEPS:
            timeslices = batch.timeslices(1)
            for t, d in zip(timeslices, distill_ranks):
                self._add_single_batch(t, d, **kwargs)

        elif self.storage_unit == StorageUnit.SEQUENCES:
            timestep_count = 0
            for seq_len in batch.get(SampleBatch.SEQ_LENS):
                start_seq = timestep_count
                end_seq = timestep_count + seq_len
                self._add_single_batch(batch[start_seq:end_seq], **kwargs)
                timestep_count = end_seq

        elif self.storage_unit == StorageUnit.EPISODES:
            for eps in batch.split_by_episode():
                if (
                    eps.get(SampleBatch.T, [0])[0] == 0
                    and eps.get(SampleBatch.DONES, [True])[-1] == True  # noqa E712
                ):
                    # Only add full episodes to the buffer
                    # Check only if info is available
                    self._add_single_batch(eps, **kwargs)
                else:
                    if log_once("only_full_episodes"):
                        logger.info(
                            "This buffer uses episodes as a storage "
                            "unit and thus allows only full episodes "
                            "to be added to it. Some samples may be "
                            "dropped."
                        )

        elif self.storage_unit == StorageUnit.FRAGMENTS:
            self._add_single_batch(batch, **kwargs)

    @DeveloperAPI
    def _add_single_batch(self, item: SampleBatchType, **kwargs) -> None:
        """Add a SampleBatch of experiences to self._storage.

        An item consists of either one or more timesteps, a sequence or an
        episode. Differs from add() in that it does not consider the storage
        unit or type of batch and simply stores it.

        Args:
            item: The batch to be added.
            ``**kwargs``: Forward compatibility kwargs.
        """
        self._num_timesteps_added += item.count
        self._num_timesteps_added_wrap += item.count
        distill_rank = item["distill_ranks"][0]

        if self._next_idx >= len(self._storage):
            self._storage.append(item)
            self._distill_index_map.setdefault(distill_rank, self._next_idx)
            self._est_size_bytes += item.size_bytes()
        else:                        
            if distill_rank >= self._distill_index_map.peekitem(0)[0]:
                self._hit = True
                _, item_to_be_removed_idx = self._distill_index_map.popitem(0)
                item_to_be_removed = self._storage[item_to_be_removed_idx]
                self._storage[item_to_be_removed_idx] = item
                self._distill_index_map.setdefault(distill_rank, item_to_be_removed_idx)
                self._est_size_bytes -= item_to_be_removed.size_bytes()
                self._est_size_bytes += item.size_bytes()
            else:
                self._hit = False

        # Eviction of older samples has already started (buffer is "full").
        if self._eviction_started & self._hit:        
            self._evicted_hit_stats.push(self._hit_count[item_to_be_removed_idx])
            self._hit_count[item_to_be_removed_idx] = 0
            self._hit = False

        # Wrap around storage as a circular buffer once we hit capacity.
        if self._num_timesteps_added_wrap >= self.capacity:
            self._eviction_started = True
            self._num_timesteps_added_wrap = 0
            self._next_idx = 0
        else:
            self._next_idx += 1