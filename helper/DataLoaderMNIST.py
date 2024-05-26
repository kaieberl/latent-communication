import torchvision.datasets as datasets
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from torch.utils.data import Dataset, Subset
import numpy as np

class DataLoader_MNIST():
    def __init__(self, batch_size, transformation1, transformation2):
        self.batch_size = batch_size
        transform1 = transforms.Compose(transformation1)
        transform2 = transforms.Compose(transformation2)

        dataset_class = datasets.MNIST
        train1 = dataset_class(root='../data', train=True, download=True, transform=transform1)
        train2 = dataset_class(root='../data', train=True, download=True, transform=transform2)
        test1 = dataset_class(root='../data', train=False, download=True, transform=transform1)
        test2 = dataset_class(root='../data', train=False, download=True, transform=transform2)

        np.random.seed(0)
        indices = np.random.permutation(len(train1))

        # Create Subsets with the same order of indices
        subset1 = Subset(train1, indices)
        subset2 = Subset(train2, indices)

        # Create DataLoaders
        self.train_loader1 = DataLoader(subset1, batch_size=batch_size, shuffle=False, num_workers=0)
        self.train_loader2 = DataLoader(subset2, batch_size=batch_size, shuffle=False, num_workers=0)
        self.test_loader1 = DataLoader(test1, batch_size=batch_size, shuffle=False, num_workers=0)
        self.test_loader2 = DataLoader(test2, batch_size=batch_size, shuffle=False, num_workers=0)
    
    def get_train_loader(self):
        return self.train_loader1, self.train_loader2
    
    def get_test_loader(self):
        return self.test_loader1, self.test_loader2