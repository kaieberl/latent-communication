import torchvision.datasets as datasets
from torch.utils.data import DataLoader
import torchvision.transforms as transforms

class DataLoader_MNIST:
    def __init__(self, batch_size, transformations):
 
        self.batch_size = batch_size
        transform = transforms.Compose(transformations)
        dataset_class = getattr(datasets, 'MNIST')
        train = dataset_class(root='../data', train=True, download=True, transform=transform)
        test = dataset_class(root='../data', train=False, download=True, transform=transform)
        self.train_loader = DataLoader(train, batch_size=self.batch_size, shuffle=True, num_workers=0)
        self.test_loader = DataLoader(test, batch_size=self.batch_size, shuffle=True, num_workers=0)
    
    def get_train_loader(self):
        return self.train_loader
    def get_test_loader(self):
        return self.test_loader
    def get_item(self, idx):
        return self.train_loader.dataset[idx]
