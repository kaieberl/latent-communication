"""This is a work in progress."""

import torch
import lightning as L
from lightning import Trainer

from models.definitions.resnet_ae import ResnetAE
from models.definitions.resnet_vae import ResnetVAE
from models.definitions.vit import MNISTDataModule


if __name__ == '__main__':
    seeds = [0, 1, 2]

    for seed in seeds:
        L.seed_everything(seed)

        data_module = MNISTDataModule(data_dir="../data", batch_size=64)
        model = ResnetAE(1, 512)    # in_channels=1 for MNIST
        trainer = Trainer(max_epochs=10)
        trainer.fit(model, datamodule=data_module)

        torch.save(model.state_dict(), f'resnet_ae_mnist_seed{seed}.pth')