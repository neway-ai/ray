import logging
from typing import Union
from sortedcontainers import SortedDict

from ray.rllib.utils.replay_buffers.replay_buffer import (
    ReplayBuffer,
    StorageUnit,
)
from ray.util.annotations import DeveloperAPI
from ray.rllib.utils.typing import SampleBatchType

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
