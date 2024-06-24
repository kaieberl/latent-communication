"""Precompute latents for all images in the given dataset.
This speeds up the optimization process by avoiding the need to compute the latents for each image in the dataset during optimization.

Usage:
    python precompute_latents.py --config-name config_name
"""
from pathlib import Path

import numpy as np
import torch
from omegaconf import DictConfig
import hydra
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST, CIFAR10
from torchvision.transforms import transforms

from utils.model import get_transformations, load_model

device = torch.device('cuda') if torch.cuda.is_available() else 'cpu'


def compute_latents(cfg, train=False):
    """Load the models and sample the latent vectors for all images in the dataset.

    Args:
        cfg (DictConfig): Configuration dictionary. Should contain a 'train_latents_path' or 'test_latents_path' key for each model.
        train (bool): If True, load the train latent vectors, defaults to test set

    Returns:
        dict: Dictionary containing the latent vectors for both models
        torch.Tensor: Labels
    """
    latents = {}
    for model_name in ['model1', 'model2']:
        if cfg.dataset == 'mnist':
            in_channels = 1
            size = 7
        elif cfg.dataset == 'cifar10':
            in_channels = 3
            size = 8
        else:
            raise ValueError("Invalid dataset")
        model = load_model(cfg[model_name].name, cfg[model_name].path, in_channels, size)
        transformations = get_transformations(cfg[model_name].name)
        transformations = transforms.Compose(transformations)
        dataset = MNIST(root=cfg.base_dir / 'data', train=train, transform=transformations, download=True) if cfg.dataset == 'mnist' else CIFAR10(root=cfg.base_dir / 'data', train=train, transform=transformations, download=True)
        dataloader = DataLoader(dataset, batch_size=128, shuffle=False)
        latents[model_name], labels = model.get_latent_space_from_dataloader(dataloader)
    return latents, labels


@hydra.main(version_base="1.1", config_path='../config')
def main(cfg: DictConfig) -> None:
    cfg.base_dir = Path(hydra.utils.get_original_cwd()).parent
    latents, labels = compute_latents(cfg)
    torch.save(labels, cfg.test_label_path)
    torch.save(latents['model1'], cfg.model1.test_latents_path)
    torch.save(latents['model2'], cfg.model2.test_latents_path)

    # latents, labels = compute_latents(cfg, train=True)
    # labels = labels.cpu().numpy()
    # np.save(cfg.train_label_path, labels)
    # torch.save(latents['model1'], cfg.model1.train_latents_path)
    # torch.save(latents['model2'], cfg.model2.train_latents_path)


if __name__ == "__main__":
    main()
