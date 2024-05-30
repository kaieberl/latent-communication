import torchvision.datasets as datasets
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from torch.utils.data import Dataset, Subset
import numpy as np

class DataLoader_MNIST():
    def __init__(self, batch_size, transformation1):
        self.batch_size = batch_size
        transform1 = transforms.Compose(transformation1)

        dataset_class = datasets.MNIST
        train1 = dataset_class(root='../data', train=True, download=True, transform=transform1)
        test1 = dataset_class(root='../data', train=False, download=True, transform=transform1)

        np.random.seed(0)
        indices = np.random.permutation(len(train1))

        # Create Subsets with the same order of indices
        subset1 = Subset(train1, indices)

        # Create DataLoaders
        self.train_loader = DataLoader(subset1, batch_size=batch_size, shuffle=False, num_workers=0)
        self.test_loader = DataLoader(test1, batch_size=batch_size, shuffle=False, num_workers=0)
    
    def get_train_loader(self):
        return self.train_loader
    def get_test_loader(self):
        return self.test_loader