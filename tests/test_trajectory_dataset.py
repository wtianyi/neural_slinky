import math
import pickle
import unittest
from priority_memory import batch

import neural_slinky.trajectory_dataset as trajectory_dataset
from torch.utils import data


class TestTrojectoryDataset(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        with open("tests/data/VanDerPol.pkl", "rb") as f:
            vdp_data = pickle.load(f)
        cls.data = vdp_data[0]["data"]
        cls.default_input_length = 1
        cls.default_target_length = 10
        cls.default_dataset = trajectory_dataset.TrajectoryDataset(
            cls.data, cls.default_input_length, cls.default_target_length
        )
    
    def check_batch_shape(self, batch, input_length, target_length):
        self.assertEqual(batch[0]["time"].shape[1], input_length)
        self.assertEqual(batch[0]["state"].shape[1], input_length)
        self.assertEqual(batch[1]["time"].shape[1], target_length)
        self.assertEqual(batch[1]["state"].shape[1], target_length)

    def test_ramdom_dataloader_creation(self, dataset=None):
        if dataset is None:
            dataset = self.default_dataset
        dataloader = trajectory_dataset.create_dataloader(dataset, batch_size=10, drop_last=True)
        for batch_size in [1, 3, 5]:
            for drop_last in [True, False]:
                dataloader = trajectory_dataset.create_dataloader(
                    dataset, batch_size=batch_size, drop_last=drop_last
                )
                counter = 0
                for d in dataloader:
                    counter += 1
                if drop_last:
                    self.assertEqual(counter, math.floor(len(dataset) / batch_size))
                    self.assertEqual(len(d[0]), batch_size)
                    self.check_batch_shape(d, dataset.input_length, dataset.target_length)
                else:
                    self.check_batch_shape(d, dataset.input_length, dataset.target_length)
                    self.assertEqual(counter, math.ceil(len(dataset) / batch_size))

    def test_stratified_sampler(self, dataset=None):
        if dataset is None:
            dataset = self.default_dataset
        for num_batches in [100]:
            for batch_size in [1, 3, 5]:
                dataloader = trajectory_dataset.create_dataloader(
                    dataset,
                    sampler_cls=trajectory_dataset.StratifiedClipSampler,
                    batch_size=batch_size,
                    drop_last=True
                )

                counter = 0
                for d in dataloader:
                    counter += 1
                    if counter >= num_batches:
                        break
                self.assertEqual(len(d[0]), batch_size)
                self.check_batch_shape(d, dataset.input_length, dataset.target_length)

    def test_varying_length(self):
        dataset = self.default_dataset
        for input_length in [1, 3, 5]:
            for target_length in [5, 10, 20]:
                dataset.input_length = input_length
                dataset.target_length = target_length
                self.test_ramdom_dataloader_creation(dataset)
                self.test_stratified_sampler(dataset)
