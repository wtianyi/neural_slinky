#!/usr/bin/env python3

import unittest
from ..batch import Batch
from ..prio import PrioritizedReplayBuffer


import torch

class TestPrio(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.device = "cuda:0"
        cls.num_data = 100000
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
