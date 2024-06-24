"""Invoke with:
    python train.py --config-name config_train -m seed=1,2,3 name=pcktae,vae latent_size=10,30,50
"""

import os.path
from pathlib import Path

import hydra
import hydra.core.global_hydra
import torch
import torch.utils.data
import lightning as L
from lightning import Trainer
from torchvision.datasets import MNIST, CIFAR10, FashionMNIST
from torchvision.transforms import Compose

from utils.model import get_transformations, load_model

hydra.core.global_hydra.GlobalHydra.instance().clear()


class MNISTDataModule(L.LightningDataModule):
    def __init__(self, transformations, data_dir=Path(os.path.realpath(__file__)).parent.parent / "data", batch_size=64):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size

        # Define transformations (same as before)
        self.transform = transformations

    def setup(self, stage=None):
        MNIST(root=self.data_dir, train=stage == "fit", download=True, transform=self.transform)

    def train_dataloader(self):
        return torch.utils.data.DataLoader(MNIST(root=self.data_dir, train=True, transform=self.transform),
                                           batch_size=self.batch_size, shuffle=True)

    def val_dataloader(self):
        return torch.utils.data.DataLoader(MNIST(root=self.data_dir, train=False, transform=self.transform),
                                           batch_size=self.batch_size, shuffle=False)


class CIFAR10DataModule(L.LightningDataModule):
    def __init__(self, transformations, data_dir=Path(os.path.realpath(__file__)).parent.parent / "data", batch_size=64):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size

        self.transform = transformations

    def setup(self, stage=None):
        CIFAR10(root=self.data_dir, train=stage == "fit", download=True, transform=self.transform)

    def train_dataloader(self):
        return torch.utils.data.DataLoader(CIFAR10(root=self.data_dir, train=True, transform=self.transform),
                                           batch_size=self.batch_size, shuffle=True)

    def val_dataloader(self):
        return torch.utils.data.DataLoader(CIFAR10(root=self.data_dir, train=False, transform=self.transform),
                                           batch_size=self.batch_size, shuffle=False)


class FMNISTDataModule(L.LightningDataModule):
    def __init__(self, transformations, data_dir=Path(os.path.realpath(__file__)).parent.parent / "data", batch_size=64):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size

        self.transform = transformations

    def setup(self, stage=None):
        FashionMNIST(root=self.data_dir, train=stage == "fit", download=True, transform=self.transform)

    def train_dataloader(self):
        return torch.utils.data.DataLoader(FashionMNIST(root=self.data_dir, train=True, transform=self.transform),
                                           batch_size=self.batch_size, shuffle=True)

    def val_dataloader(self):
        return torch.utils.data.DataLoader(FashionMNIST(root=self.data_dir, train=False, transform=self.transform),
                                           batch_size=self.batch_size, shuffle=False)


@hydra.main(config_path="../config")
def main(cfg):
    base_dir = Path(hydra.utils.get_original_cwd()).parent

    L.seed_everything(cfg.seed)

    transformations = Compose(get_transformations(cfg.name))
    if cfg.dataset == "cifar10":
        data_module = CIFAR10DataModule(transformations)
        model = load_model(cfg.name, in_channels=3, size=8)
    elif cfg.dataset == "mnist":
        data_module = MNISTDataModule(transformations)
        model = load_model(cfg.name, in_channels=1)
    elif cfg.dataset == "fmnist":
        data_module = FMNISTDataModule(transformations)
        model = load_model(cfg.name, in_channels=1)
    else:
        raise ValueError(f"Unknown dataset: {cfg.dataset}")
    trainer = Trainer(max_epochs=cfg.epochs)
    trainer.fit(model, datamodule=data_module)

    path = base_dir / "models" / f"{cfg.dataset.upper()}_{cfg.name.upper()}_{cfg.latent_size}_{cfg.seed}.pth"
    torch.save(model.state_dict(), path)


if __name__ == '__main__':
    main()