from pathlib import Path

import torch
from omegaconf import DictConfig
import hydra
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST
from torchvision.transforms import transforms

from stitching.stitching import get_transformations, load_model
from optimizer import AffineFitting, LinearFitting
from utils.sampler import simple_sampler
from utils.visualization import visualize_results

device = torch.device('cuda') if torch.cuda.is_available() else 'cpu'


def get_latents(cfg, test=False):
    """For both models, load the latent vectors if latent_path is provided, else load the models and sample the
        latent vectors.

    Args:
        cfg (DictConfig): Configuration dictionary
        test (bool): If True, load the test latent vectors

    Returns:
        dict: Dictionary containing the latent vectors for both models
        torch.Tensor: Labels
    """
    latents = {}
    if not test:
        torch.manual_seed(0)
        indices = torch.randperm(60000)[:cfg.num_samples]
        for model_name in ['model1', 'model2']:
            if 'latents_path' in cfg[model_name]:
                labels = torch.load(cfg[model_name].train_label_path, map_location=device)
                z = torch.load(cfg[model_name].train_latents_path, map_location=device)
                latents[model_name] = z[indices]
            else:
                model = load_model(cfg[model_name].name, cfg[model_name].path)
                transformations = get_transformations(cfg[model_name].name)
                latents[model_name], labels = simple_sampler(indices, model, transformations, device, seed=cfg[model_name].seed)
    else:
        for model_name in ['model1', 'model2']:
            if 'latents_path' in cfg[model_name]:
                labels = torch.load(cfg[model_name].test_label_path, map_location=device)
                z = torch.load(cfg[model_name].test_latents_path, map_location=device)
                latents[model_name] = z
            else:
                model = load_model(cfg[model_name].name, cfg[model_name].path)
                transformations = get_transformations(cfg[model_name].name)
                transformations = transforms.Compose(transformations)
                dataset = MNIST(root=cfg.base_dir / 'data', train=False, transform=transformations, download=True)
                dataloader = DataLoader(dataset, batch_size=128, shuffle=False)
                latents[model_name], labels = model.get_latent_space_from_dataloader(dataloader)
    return latents, labels


@hydra.main(config_path="../config", config_name="config_resnet")
def main(cfg : DictConfig) -> None:
    cfg.base_dir = Path(hydra.utils.get_original_cwd()).parent
    latents1, latents2 = get_latents(cfg)[0].values()

    # Linear transform
    linear_fitting = LinearFitting(latents1, latents2, lamda=0.01)
    linear_fitting.solve_problem()
    storage_path = Path(cfg.storage_path) / f"Linear_{cfg.model1.name}_{cfg.model1.seed}_{cfg.model2.name}_{cfg.model2.seed}_{cfg.num_samples}"
    linear_fitting.save_results(storage_path)

    latents, labels = get_latents(cfg, test=True)
    latents1, latents2 = latents.values()
    _, A = linear_fitting.get_results()
    latents1_trafo = latents1 @ A.T
    visualize_results(cfg, labels, latents2, latents1_trafo)
    del linear_fitting

    # Affine transform
    affine_fitting = AffineFitting(latents1, latents2, lamda=0.01)
    affine_fitting.solve_problem()
    storage_path = Path(cfg.storage_path) / f"Affine_{cfg.model1.name}_{cfg.model1.seed}_{cfg.model2.name}_{cfg.model2.seed}_{cfg.num_samples}"
    affine_fitting.save_results(storage_path)

    _, A, b = affine_fitting.get_results()
    latents1_trafo = latents1 @ A.T + b
    visualize_results(cfg, labels, latents2, latents1_trafo)


if __name__ == '__main__':
    main()
