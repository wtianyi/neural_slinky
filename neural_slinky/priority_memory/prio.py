#!/usr/bin/env python3

# Modified from https://github.com/thu-ml/tianshou/blob/master/tianshou/data/buffer/prio.py
from typing import Any, List, Optional, Tuple, Union, Dict

import numpy as np
import torch

from copy import deepcopy
from numbers import Number

from .batch import Batch, _parse_value, _alloc_by_keys_diff, _create_value
from .segtree import SegmentTree


def to_numpy(x: Any) -> Union[Batch, np.ndarray]:
    """Return an object without torch.Tensor."""
    if isinstance(x, torch.Tensor):  # most often case
        return x.detach().cpu().numpy()
    elif isinstance(x, np.ndarray):  # second often case
        return x
    elif isinstance(x, (np.number, np.bool_, Number)):
        return np.asanyarray(x)
    elif x is None:
        return np.array(None, dtype=object)
    elif isinstance(x, (dict, Batch)):
        x = Batch(x) if isinstance(x, dict) else deepcopy(x)
        x.to_numpy()
        return x
    elif isinstance(x, (list, tuple)):
        return to_numpy(_parse_value(x))
    else:  # fallback
        return np.asanyarray(x)


class PrioritizedReplayBuffer:
    """Implementation of Prioritized Experience Replay. arXiv:1511.05952.
    :param float alpha: the prioritization exponent.
    :param float beta: the importance sample soft coefficient.
    :param bool weight_norm: whether to normalize returned weights with the maximum
        weight value within the batch. Default to True.
    .. seealso::
        Please refer to :class:`~tianshou.data.ReplayBuffer` for other APIs' usage.
    """

    def __init__(
        self,
        size: int,
        alpha: float,
        beta: float,
        weight_norm: bool = True,
        **kwargs: Any
    ) -> None:
        # will raise KeyError in PrioritizedVectorReplayBuffer
        # super().__init__(size, **kwargs)
        self.maxsize = size

        self._meta: Batch = Batch()
        self._indices = np.arange(size)

        assert alpha > 0.0 and beta >= 0.0
        self._alpha, self._beta = alpha, beta
        self._max_prio = self._min_prio = 1.0
        # save weight directly in this class instead of self._meta
        self.weight = SegmentTree(size)
        self.__eps = np.finfo(np.float32).eps.item()
        self.options: Dict[str, Any] = dict(alpha=alpha, beta=beta)
        self._weight_norm = weight_norm
        self.reset()

    def __len__(self) -> int:
        """Return len(self)."""
        return self._size

    def __repr__(self) -> str:
        """Return str(self)."""
        return self.__class__.__name__ + self._meta.__repr__()[5:]

    def __getattr__(self, key: str) -> Any:
        """Return self.key."""
        try:
            return self._meta[key]
        except KeyError as e:
            raise AttributeError from e

    def reset(self) -> None:
        """Clear all the data in replay buffer."""
        self.last_index = np.array([0])
        self._index = self._size = 0

    def init_weight(self, index: Union[int, np.ndarray]) -> None:
        self.weight[index] = self._max_prio ** self._alpha

    # def update(self, buffer: ReplayBuffer) -> np.ndarray:
    #     indices = super().update(buffer)
    #     self.init_weight(indices)
    #     return indices

    def __getitem_from_meta__(
        self, index: Union[slice, int, List[int], np.ndarray]
    ) -> Batch:
        """Return a data batch: self[index].
        If stack_num is larger than 1, return the stacked obs and obs_next with shape
        (batch, len, ...).
        """
        if isinstance(index, slice):  # change slice to np array
            # buffer[:] will get all available data
            indices = (
                self.sample_indices(0)
                if index == slice(None)
                else self._indices[: len(self)][index]
            )
        else:
            indices = index
        # raise KeyError first instead of AttributeError,
        # to support np.array([ReplayBuffer()])
        return self._meta[indices]

    def _add_index(self) -> int:
        """Maintain the buffer's state after adding one data batch.
        Returns:
            index_to_be_modified
        """
        self.last_index[0] = ptr = self._index
        self._size = min(self._size + 1, self.maxsize)
        self._index = (self._index + 1) % self.maxsize
        return ptr

    def add(
        self,
        batch: Batch,
        # buffer_ids: Optional[Union[np.ndarray, List[int]]] = None
    ) -> int:
        """Add a batch of data into replay buffer.
        :param Batch batch: the input data batch

        Returns:
            current_index
        """
        # ptr, ep_rew, ep_len, ep_idx = super().add(batch, buffer_ids)

        ptr = self._add_index()

        try:
            self._meta[ptr] = batch
        except ValueError:
            if self._meta.is_empty():
                self._meta = _create_value(  # type: ignore
                    batch, self.maxsize
                )
            else:  # dynamic key pops up in batch
                _alloc_by_keys_diff(self._meta, batch, self.maxsize)
            self._meta[ptr] = batch

        self.init_weight(ptr)
        return ptr

    def sample_indices(self, batch_size: int) -> np.ndarray:
        if batch_size > 0 and len(self) > 0:
            scalar = np.random.rand(batch_size) * self.weight.reduce()
            return self.weight.get_prefix_sum_idx(scalar)  # type: ignore
        else:
            if batch_size > 0:
                return np.random.choice(self._size, batch_size)
            elif batch_size == 0:  # construct current available indices
                return np.concatenate(
                    [np.arange(self._index, self._size), np.arange(self._index)]
                )
            else:
                return np.array([], int)
            # return super().sample_indices(batch_size)

    def sample(self, batch_size: int) -> Tuple[Batch, np.ndarray]:
        """Get a random sample from buffer with size = batch_size.
        Return all the data in the buffer if batch_size is 0.
        :return: Sample data and its corresponding index inside the buffer.
        """
        indices = self.sample_indices(batch_size)
        return self[indices], indices

    def get_weight(self, index: Union[int, np.ndarray]) -> Union[float, np.ndarray]:
        """Get the importance sampling weight.
        The "weight" in the returned Batch is the weight on loss function to debias
        the sampling process (some transition tuples are sampled more often so their
        losses are weighted less).
        """
        # important sampling weight calculation
        # original formula: ((p_j/p_sum*N)**(-beta))/((p_min/p_sum*N)**(-beta))
        # simplified formula: (p_j/p_min)**(-beta)
        return (self.weight[index] / self._min_prio) ** (-self._beta)

    def update_priority(
        self, index: np.ndarray, new_weight: Union[np.ndarray, torch.Tensor]
    ) -> None:
        """Update priority weight by index in this buffer.
        :param np.ndarray index: index you want to update weight.
        :param np.ndarray new_weight: new priority weight you want to update.
        """
        weight = np.abs(to_numpy(new_weight)) + self.__eps
        self.weight[index] = weight ** self._alpha
        self._max_prio = max(self._max_prio, weight.max())
        self._min_prio = min(self._min_prio, weight.min())

    def __getitem__(self, index: Union[slice, int, List[int], np.ndarray]) -> Batch:
        if isinstance(index, slice):  # change slice to np array
            # buffer[:] will get all available data
            indices = (
                self.sample_indices(0)
                if index == slice(None)
                else self._indices[: len(self)][index]
            )
        else:
            indices = index
        batch = self.__getitem_from_meta__(indices)
        weight = self.get_weight(indices)
        # ref: https://github.com/Kaixhin/Rainbow/blob/master/memory.py L154
        batch.weight = weight / np.max(weight) if self._weight_norm else weight
        return batch

    def set_beta(self, beta: float) -> None:
        self._beta = beta


class PrioritizedReplayBetaScheduler(object):
    def __init__(
        self, pr_buf: PrioritizedReplayBuffer, T_max: int, end_beta: float = 1
    ):
        self.pr_buf = pr_buf
        self.beta = pr_buf._beta
        self.T_max = T_max
        self.end_beta = end_beta
        self.step_size = (end_beta - self.beta) / T_max

    def step(self):
        self.beta = min(1.0, self.beta + self.step_size)
        self.pr_buf.set_beta(self.beta)
