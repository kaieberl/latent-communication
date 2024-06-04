import os
from pathlib import Path

import torchvision.datasets as datasets
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from torch.utils.data import Subset
import numpy as np


class DataLoaderMNIST:
    def __init__(self, batch_size, transformation, seed=0, indices=None, base_path=os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))):
        self.indices = indices
        self.seed = seed
        self.batch_size = batch_size
        transform = transforms.Compose(transformation)

        dataset_class = datasets.MNIST
        train = dataset_class(root=base_path + '/data', train=True, download=True, transform=transform)
        test = dataset_class(root=base_path + '/data', train=False, download=True, transform=transform)

        np.random.seed(self.seed)

        # Create Subsets with the same order of indices
        if self.indices is None:
            self.indices = np.random.permutation(len(train))
        subset = Subset(train, self.indices)

        # Create DataLoaders
        self.train_loader = DataLoader(subset, batch_size=batch_size, shuffle=False, num_workers=0)
        self.test_loader = DataLoader(test, batch_size=batch_size, shuffle=False, num_workers=0)

    def get_train_loader(self):
        return self.train_loader

    def get_test_loader(self):
        return self.test_loader
