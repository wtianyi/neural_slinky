from typing import Iterator, Optional, Sequence, Union

import numpy as np
from torch.utils import data

from priority_memory import batch


class TrajectoryDataset(data.Dataset[data.dataset.T_co]):
    def __init__(
        self,
        data: Sequence[data.dataset.T_co],
        input_length: int = 1,
        target_length: int = 1,
    ):
        """Dataset for time-series data that supports indexing of fixed-length subsequence

        A clip is an input-output pair of subarray from a trajectory.

        Args:
            data (Sequence[data.dataset.T_co]): A list of trajectories
            traj_stratified (bool, optional): Whether to stratify (based on trajectory lengths, so that longer trajectories get more samples) when sampling clips. Defaults to False.
        """
        super().__init__()
        self._input_length = input_length
        self._target_length = target_length
        self._length_list = [len(d) for d in data]
        self.num_trajectories = len(self._length_list)
        self._internal_start_inds = np.cumsum([0] + self._length_list)[:-1]
        self._data = batch.Batch.cat([batch.Batch(d) for d in data])
        self._calculate_ind_mapping()

    def __len__(self) -> int:
        return np.sum(self._valid_len_list)

    @property
    def input_length(self):
        return self._input_length

    @input_length.setter
    def input_length(self, input_length: int):
        self._input_length = input_length
        self._calculate_ind_mapping()

    @property
    def target_length(self):
        return self._target_length

    @target_length.setter
    def target_length(self, target_length: int):
        self._target_length = target_length
        self._calculate_ind_mapping()

    @staticmethod
    def _make_range_indices(
        begin: Union[int, np.integer, np.ndarray],
        end: Union[int, np.integer, np.ndarray],
    ):
        if isinstance(begin, (int, np.integer)):
            return np.arange(begin, end)
        return np.stack([np.arange(a, b) for a, b in zip(begin, end)])

    def __getitem__(self, index):
        # if isinstance(index, int):
        #     index = [index]  # to preserve the temporal axis even if it has size 1
        start_inds = self._ind_map[index]
        input_end_inds = start_inds + self._input_length
        target_end_inds = input_end_inds + self._target_length
        input = self._data[self._make_range_indices(start_inds, input_end_inds), ...]
        target = self._data[
            self._make_range_indices(input_end_inds, target_end_inds), ...
        ]
        return input, target

    def _calculate_ind_mapping(self) -> int:
        """Calculate the valid range of starting temporal index for clip samples

        Sets self._valid_interval_list is a list of upper_idx (int) so that `np.random.randint(upper_idx)` will give a valid sample for starting temporal index
        """
        self._valid_len_list = np.array(
            [
                l - self._input_length - self._target_length + 1
                for l in self._length_list
            ]
        )
        self._extern_start_inds = np.cumsum(np.insert(self._valid_len_list, 0, 0))
        # self._traj_weights = self._valid_len_list / self._valid_len_list.sum()
        self._ind_map = np.arange(len(self))
        for i in range(len(self._extern_start_inds) - 1):
            start_ind = self._extern_start_inds[i]
            end_ind = self._extern_start_inds[i + 1]
            self._ind_map[start_ind:end_ind] += self._internal_start_inds[i] - start_ind

    def to_dataloader(
        self, batch_size: int, sampler: str, drop_last: bool = True, **kwargs
    ):
        """Get dataloader from dataset.

        Args:
            batch_size (int): batch size
            sampler (str): The type of sampler. Can be "sequential", "random" or
                           "stratified"
            drop_last (bool, optional): drop_last. Defaults to True.

        Raises:
            ValueError: Unknown sampler type

        Returns:
            data.DataLoader: dataloader
        """
        if sampler == "random":
            sampler_cls = data.RandomSampler
        elif sampler == "stratified":
            sampler_cls = StratifiedClipSampler
        elif sampler == "sequential":
            sampler_cls = data.SequentialSampler
        else:
            raise ValueError(f"Unknown sampler type: {sampler}")
        return create_dataloader(
            self, batch_size, drop_last, sampler_cls=sampler_cls, **kwargs
        )


class StratifiedClipSampler(data.RandomSampler):
    def __init__(
        self,
        data_source: TrajectoryDataset,
        num_samples: Optional[int] = None,
    ) -> None:
        super(StratifiedClipSampler, self).__init__(data_source, True, num_samples)

    def _sample_traj_idx(self, n: int = 1) -> np.ndarray:
        return np.random.randint(self.data_source.num_trajectories, size=n)

    def _sample_clip_from_traj(self, traj_idx: np.ndarray):
        in_traj_ind = np.random.randint(self.data_source._valid_len_list[traj_idx])
        return in_traj_ind + self.data_source._extern_start_inds[traj_idx]

    def __iter__(self) -> Iterator[int]:
        for _ in range(self.num_samples // 32):
            yield from self._sample_clip_from_traj(self._sample_traj_idx(32))
        yield from self._sample_clip_from_traj(
            self._sample_traj_idx(self.num_samples % 32)
        )


def create_dataloader(
    dataset: TrajectoryDataset,
    batch_size: int,
    drop_last: bool,
    sampler_cls=data.RandomSampler,
    **kw_args,
):
    # specify the `sampler` argument to suppress the `auto_collation` of DataLoader
    return data.DataLoader(
        dataset,
        sampler=data.BatchSampler(
            sampler_cls(dataset),
            batch_size=batch_size,
            drop_last=drop_last,
        ),
        batch_size=None,
        **kw_args,
    )
