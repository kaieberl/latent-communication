import os
from pathlib import Path  #
import torch

import torchvision.datasets as datasets
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from torch.utils.data import Subset
import numpy as np


class DataLoaderMNIST:
    def __init__(self, batch_size, transformation, seed=0, indices=None,
                 base_path=Path(os.path.realpath(__file__)).parent.parent.parent, shuffle_train_flag=False):
        self.indices = indices
        self.seed = seed
        self.batch_size = batch_size
        transform = transforms.Compose(transformation)

        dataset_class = datasets.MNIST
        self.train_dataset = dataset_class(root=base_path / 'data', train=True, download=False, transform=transform)
        self.test_dataset = dataset_class(root=base_path / 'data', train=False, download=True, transform=transform)

        np.random.seed(self.seed)

        # Create Subsets with the same order of indices
        if self.indices is None:
            self.indices = np.random.permutation(len(self.train_dataset))
        self.train_subset = Subset(self.train_dataset, self.indices)

        # Create DataLoaders
        self.train_loader = DataLoader(self.train_subset, batch_size=batch_size, shuffle=shuffle_train_flag,
                                       num_workers=0)
        self.test_loader = DataLoader(self.test_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

        #self.input_size = self.train_dataset[0][0].shape[1]

    def get_train_loader(self):
        return self.train_loader

    def get_test_loader(self):
        return self.test_loader

    def get_full_train_dataset(self):
        """
        Returns the full training dataset as a tensor.
        """
        train_data = []
        train_labels = []
        for data, label in self.train_dataset:
            train_data.append(data.numpy())
            train_labels.append(label)
        return torch.tensor(train_data), torch.tensor(train_labels)

    def get_full_test_dataset(self):
        """
        Returns the full test dataset as a tensor.
        """
        test_data = []
        test_labels = []
        for data, label in self.test_dataset:
            test_data.append(data.numpy())
            test_labels.append(label)
        return torch.tensor(test_data), torch.tensor(test_labels)

    def get_train_subset(self):
        """
        Returns the training subset as a tensor.
        """
        subset_data = []
        subset_labels = []
        for idx in self.train_subset.indices:
            data, label = self.train_dataset[idx]
            subset_data.append(data.numpy())
            subset_labels.append(label)
        return torch.tensor(subset_data), torch.tensor(subset_labels)


class DataLoaderFashionMNIST:
    def __init__(self, batch_size, transformation, seed=0, indices=None,
                 base_path=Path(os.path.realpath(__file__)).parent.parent.parent, shuffle_train_flag=False):
        self.indices = indices
        self.seed = seed
        self.batch_size = batch_size
        transform = transforms.Compose(transformation)

        dataset_class = datasets.FashionMNIST
        self.train_dataset = dataset_class(root=base_path / 'data', train=True, download=True, transform=transform)
        self.test_dataset = dataset_class(root=base_path / 'data', train=False, download=True, transform=transform)

        np.random.seed(self.seed)

        # Create Subsets with the same order of indices
        if self.indices is None:
            self.indices = np.random.permutation(len(self.train_dataset))
        self.train_subset = Subset(self.train_dataset, self.indices)

        # Create DataLoaders
        self.train_loader = DataLoader(self.train_subset, batch_size=batch_size, shuffle=shuffle_train_flag,
                                       num_workers=0)
        self.test_loader = DataLoader(self.test_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

        #self.input_size = self.train_dataset[0][0].shape[1]

    def get_train_loader(self):
        return self.train_loader

    def get_test_loader(self):
        return self.test_loader

    def get_full_train_dataset(self):
        """
        Returns the full training dataset as a tensor.
        """
        train_data = []
        train_labels = []
        for data, label in self.train_dataset:
            train_data.append(data.numpy())
            train_labels.append(label)
        return torch.tensor(train_data), torch.tensor(train_labels)

    def get_full_test_dataset(self):
        """
        Returns the full test dataset as a tensor.
        """
        test_data = []
        test_labels = []
        for data, label in self.test_dataset:
            test_data.append(data.numpy())
            test_labels.append(label)

        test_data = np.array(test_data)
        test_labels = np.array(test_labels)
        return torch.tensor(test_data), torch.tensor(test_labels)

    def get_train_subset(self):
        """
        Returns the training subset as a tensor.
        """
        subset_data = []
        subset_labels = []
        for idx in self.train_subset.indices:
            data, label = self.train_dataset[idx]
            subset_data.append(data.numpy())
            subset_labels.append(label)
        return torch.tensor(subset_data), torch.tensor(subset_labels)


class DataLoaderCIFAR10:
    def __init__(self, batch_size, transformation, seed=0, indices=None,
                 base_path=Path(os.path.realpath(__file__)).parent.parent.parent, shuffle_train_flag=False):
        self.indices = indices
        self.seed = seed
        self.batch_size = batch_size
        transform = transforms.Compose(transformation)

        dataset_class = datasets.CIFAR10
        self.train_dataset = dataset_class(root=base_path / 'data', train=True, download=False, transform=transform)
        self.test_dataset = dataset_class(root=base_path / 'data', train=False, download=False, transform=transform)

        np.random.seed(self.seed)

        # Create Subsets with the same order of indices
        if self.indices is None:
            self.indices = np.random.permutation(len(self.train_dataset))
        self.train_subset = Subset(self.train_dataset, self.indices)

        # Create DataLoaders
        self.train_loader = DataLoader(self.train_subset, batch_size=batch_size, shuffle=shuffle_train_flag,
                                       num_workers=0)
        self.test_loader = DataLoader(self.test_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

        self.input_size = self.train_dataset[0][0].shape[1]

    def get_train_loader(self):
        return self.train_loader

    def get_test_loader(self):
        return self.test_loader

    def get_full_train_dataset(self):
        """
        Returns the full training dataset as a tensor.
        """
        train_data = []
        train_labels = []
        for data, label in self.train_dataset:
            train_data.append(data.numpy())
            train_labels.append(label)
        return torch.tensor(train_data), torch.tensor(train_labels)

    def get_full_test_dataset(self):
        """
        Returns the full test dataset as a tensor.
        """
        test_data = []
        test_labels = []
        for data, label in self.test_dataset:
            test_data.append(data.numpy())
            test_labels.append(label)
        return torch.tensor(test_data), torch.tensor(test_labels)

    def get_train_subset(self):
        """
        Returns the training subset as a tensor.
        """
        subset_data = []
        subset_labels = []
        for idx in self.train_subset.indices:
            data, label = self.train_dataset[idx]
            subset_data.append(data.numpy())
            subset_labels.append(label)
        return torch.tensor(subset_data), torch.tensor(subset_labels)


class DataLoaderCIFAR100:
    def __init__(self, batch_size, transformation, seed=0, indices=None,
                 base_path=Path(os.path.realpath(__file__)).parent.parent.parent, shuffle_train_flag=False):
        self.indices = indices
        self.seed = seed
        self.batch_size = batch_size
        transform = transforms.Compose(transformation)

        dataset_class = datasets.CIFAR100
        self.train_dataset = dataset_class(root=base_path / 'data', train=True, download=False, transform=transform)
        self.test_dataset = dataset_class(root=base_path / 'data', train=False, download=False, transform=transform)

        np.random.seed(self.seed)

        # Create Subsets with the same order of indices
        if self.indices is None:
            self.indices = np.random.permutation(len(self.train_dataset))
        self.train_subset = Subset(self.train_dataset, self.indices)

        # Create DataLoaders
        self.train_loader = DataLoader(self.train_subset, batch_size=batch_size, shuffle=shuffle_train_flag,
                                       num_workers=0)
        self.test_loader = DataLoader(self.test_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

        self.input_size = self.train_dataset[0][0].shape[1]

    def get_train_loader(self):
        return self.train_loader

    def get_test_loader(self):
        return self.test_loader

    def get_full_train_dataset(self):
        """
        Returns the full training dataset as a tensor.
        """
        train_data = []
        train_labels = []
        for data, label in self.train_dataset:
            train_data.append(data.numpy())
            train_labels.append(label)
        return torch.tensor(train_data), torch.tensor(train_labels)

    def get_full_test_dataset(self):
        """
        Returns the full test dataset as a tensor.
        """
        test_data = []
        test_labels = []
        for data, label in self.test_dataset:
            test_data.append(data.numpy())
            test_labels.append(label)
        return torch.tensor(test_data), torch.tensor(test_labels)

    def get_train_subset(self):
        """
        Returns the training subset as a tensor.
        """
        subset_data = []
        subset_labels = []
        for idx in self.train_subset.indices:
            data, label = self.train_dataset[idx]
            subset_data.append(data.numpy())
            subset_labels.append(label)
        return torch.tensor(subset_data), torch.tensor(subset_labels)
