"""Precompute latents for all images in the given dataset.
This speeds up the optimization process by avoiding the need to compute the latents for each image in the dataset during optimization.

Usage:
    python precompute_latents.py --config-name config_name
"""

import torch
from omegaconf import DictConfig
import hydra
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST
from torchvision.transforms import transforms

from utils.model import get_transformations, load_model

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('mps') if torch.backends.mps.is_available() else 'cpu'



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
        model = load_model(cfg[model_name].name, cfg[model_name].path)
        transformations = get_transformations(cfg[model_name].name)
        transformations = transforms.Compose(transformations)
        dataset = MNIST(root=cfg.base_dir / 'data', train=train, transform=transformations, download=True)
        dataloader = DataLoader(dataset, batch_size=128, shuffle=False)
        latents[model_name], labels = model.get_latent_space_from_dataloader(dataloader)
    return latents, labels


@hydra.main(version_base="1.1", config_path='../config')
def main(cfg: DictConfig) -> None:
    cfg.base_dir = hydra.utils.get_original_cwd()
    latents, labels = compute_latents(cfg)
    torch.save(latents, cfg.latents_path)
    torch.save(labels, cfg.labels_path)
