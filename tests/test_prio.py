#!/usr/bin/env python3

import unittest

from priority_memory.batch import Batch
from priority_memory.prio import PrioritizedReplayBuffer
from priority_memory.pr_dataloader import PrioritizedReplayDataLoader
from torch.utils import data

import torch
import numpy as np
import random

from matplotlib import pyplot as plt


class TestPrio(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.device = "cuda:0"
        cls.num_data = 1000
        cls.feature_dim = 6
        cls.target_dim = 3
        cls.data = torch.randn(cls.num_data, cls.feature_dim, device=cls.device)
        cls.target = torch.randn(cls.num_data, cls.target_dim, device=cls.device)
        return

    def test_add_batch(self):
        prio_buf = PrioritizedReplayBuffer(100, 0.1, 0.1)
        for x, y in zip(self.data, self.target):
            prio_buf.add(Batch(x=x, y=y))

        batch_size = 10
        batch, indices = prio_buf.sample(batch_size)

        x_batch, y_batch = batch["x"], batch["y"]

        self.assertEqual(x_batch.shape[0], batch_size)
        priorities = torch.rand(batch_size, device=self.device)
        print(priorities)
        prio_buf.update_priority(indices, priorities)
        print(prio_buf.weight[indices])

        print(prio_buf.get_weight(indices))
        return


def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)


class TestPrioLoader(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        set_seed(0)
        cls.device = "cuda:0"
        cls.num_data = 900
        cls.feature_dim = 6
        cls.target_dim = 3
        cls.data_idx = torch.arange(cls.num_data, device=cls.device)
        cls.data = torch.randn(cls.num_data, cls.feature_dim, device=cls.device)
        cls.model_coef = torch.randn(cls.feature_dim, cls.target_dim, device=cls.device)
        cls.target = cls.data @ cls.model_coef
        noise_levels = torch.tensor([0, 0.1, 1], device=cls.device)
        noise = torch.repeat_interleave(noise_levels, cls.num_data // len(noise_levels))
        cls.target += noise[:, None] * torch.randn(
            cls.num_data, cls.target_dim, device=cls.device
        )
        return

    def test_tuple_data(self):
        set_seed(0)
        pr_buffer = PrioritizedReplayBuffer(self.num_data, 0.1, 0.1)
        dataloader = data.DataLoader(
            data.TensorDataset(self.data_idx, self.data, self.target),
            batch_size=1000,
            shuffle=False,
        )
        pr_loader = PrioritizedReplayDataLoader(
            pr_buffer=pr_buffer, dataloader=dataloader, replay_prob=0
        )
        for (idx, input_data, target), weight in pr_loader:
            weight = torch.tensor(weight[:, None], device=self.device)
            loss = ((target - input_data @ self.model_coef) * weight).mean(dim=1)
            pr_loader.update_latest_batch_score(loss)
        plt.plot(pr_buffer.get_weight(slice(None)))
        plt.savefig("test_tuple_data_weights.png")

    def test_tuple_data_with_replay(self):
        pr_buffer = PrioritizedReplayBuffer(self.num_data // 2, 0.1, 0.1)
        dataloader = data.DataLoader(
            data.TensorDataset(self.data, self.target), shuffle=False
        )
        pr_loader = PrioritizedReplayDataLoader(
            pr_buffer=pr_buffer, dataloader=dataloader, replay_prob=0.5
        )
        for (idx, input_data, target), weight in pr_loader:
            weight = torch.tensor(weight[:, None], device=self.device)
            loss = ((target - input_data @ self.model_coef) * weight).mean(dim=1)
            pr_loader.update_latest_batch_score(loss)
