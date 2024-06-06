import hydra
import torch
import torch.utils.data
import lightning as L
from lightning import Trainer
from torchvision.datasets import MNIST

from utils.model import get_transformations, load_model


class MNISTDataModule(L.LightningDataModule):
    def __init__(self, data_dir="../data", batch_size=64):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size

        # Define transformations (same as before)
        self.transform = get_transformations('resnet_ae')

    def setup(self, stage=None):
        MNIST(root=self.data_dir, train=stage == "fit", download=True, transform=self.transform)

    def train_dataloader(self):
        return torch.utils.data.DataLoader(MNIST(root=self.data_dir, train=True, transform=self.transform),
                                           batch_size=self.batch_size, shuffle=True)

    def val_dataloader(self):
        return torch.utils.data.DataLoader(MNIST(root=self.data_dir, train=False, transform=self.transform),
                                           batch_size=self.batch_size, shuffle=False)


@hydra.main(config_path="../config")
def main(cfg):
    seeds = range(3)

    for seed in seeds:
        L.seed_everything(seed)

        data_module = MNISTDataModule()
        model = load_model(cfg.model.name)
        trainer = Trainer(max_epochs=10)
        trainer.fit(model, datamodule=data_module)

        torch.save(model.state_dict(), cfg.model.path)


if __name__ == '__main__':
    main()