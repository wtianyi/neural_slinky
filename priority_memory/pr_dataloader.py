#!/usr/bin/env python3
from typing import Sequence, Tuple
from torch.utils.data import DataLoader
import numpy as np
from priority_memory.batch import Batch
from priority_memory.prio import PrioritizedReplayBuffer


class PrioritizedReplayDataLoader:
    def __init__(
        self,
        pr_buffer: PrioritizedReplayBuffer,
        dataloader: DataLoader,
        replay_prob: float = 0.5,
    ):
        self.pr_buffer = pr_buffer
        self.dataloader = dataloader
        self.replay_prob = replay_prob
        self._take_new: bool = True
        self.latest_buffer_inds = None

    def __len__(self) -> int:
        return len(self.dataloader)

    def __iter__(self):
        return PrioIter(self)

    def update_latest_batch_score(self, scores):
        self.pr_buffer.update_priority(self.latest_buffer_inds, scores)

    @property
    def take_new(self) -> bool:
        take_new = self._take_new
        return take_new

    @take_new.getter
    def take_new(self) -> bool:
        take_new = self._take_new
        self._take_new = np.random.rand() > self.replay_prob
        return take_new

    @take_new.setter
    def take_new(self, take_new: bool):
        self._take_new = take_new


class PrioIter:
    def __init__(self, loader: PrioritizedReplayDataLoader):
        self.loader = loader
        self.dataloader_iter = iter(self.loader.dataloader)
        self.latest_buffer_inds = None

    def _collator(self, batch: Batch):
        if self.type == "single":
            return batch["data"]
        elif self.type == "dict":
            return batch
        elif self.type == "tuple":
            return (batch[k] for k in self.keys)

    def __next__(self):
        if self.loader.take_new:
            data = next(self.dataloader_iter)
            # print(type(data))
            if isinstance(data, (dict, Batch)):
                self.type = "dict"
                self.keys = list(data.keys())
                self.latest_batch = Batch(data)
            elif isinstance(data, Sequence):
                self.type = "tuple"
                self.keys = [f"data_{i}" for i in range(len(data))]
                self.latest_batch = Batch({k: v for k, v in zip(self.keys, data)})
            else:
                self.type = "single"
                self.keys = ["data"]
                self.latest_batch = Batch({"data": data})
            inds = self.loader.pr_buffer.add_batch(
                self.latest_batch
            )
            # self.latest_buffer_inds = self.loader.pr_buffer.add_batch(self.latest_batch)
            self.loader.latest_buffer_inds = inds
            return data, self.loader.pr_buffer.get_weight(inds)
        else:
            inds = self.loader.pr_buffer.sample_indices(
                self.loader.dataloader.batch_size
            )
            # self.latest_buffer_inds = inds
            self.loader.latest_buffer_inds = inds
            data_batch = self.loader.pr_buffer[inds]
            data = self._collator(data_batch)
            return data, data_batch["weight"]
