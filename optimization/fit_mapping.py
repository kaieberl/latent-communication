from pathlib import Path

import torch
from omegaconf import DictConfig
import hydra
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST
from torchvision.transforms import transforms

from stitching.stitching import get_transformations, load_model
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


@hydra.main(config_path="../config", config_name="config_resnet_nn")
def main(cfg : DictConfig) -> None:
    cfg.base_dir = Path(hydra.utils.get_original_cwd()).parent
    latents1, latents2 = get_latents(cfg)[0].values()

    if cfg.mapping == 'Linear':
        from optimizer import LinearFitting
        mapping = LinearFitting(latents1, latents2, lamda=cfg.lamda)
    elif cfg.mapping == 'Affine':
        from optimizer import AffineFitting
        mapping = AffineFitting(latents1, latents2, lamda=cfg.lamda)
    elif cfg.mapping == 'NeuralNetwork':
        from optimizer import NeuralNetworkFitting
        mapping = NeuralNetworkFitting(latents1, latents2, hidden_dim=cfg.hidden_size, lamda=cfg.lamda, learning_rate=cfg.learning_rate, epochs=cfg.epochs)
    else:
        raise ValueError("Invalid experiment name")
    mapping.fit()
    storage_path = Path(cfg.storage_path) / f"{cfg.mapping}_{cfg.model1.name}_{cfg.model1.seed}_{cfg.model2.name}_{cfg.model2.seed}_{cfg.num_samples}"
    mapping.save_results(storage_path)

    latents, labels = get_latents(cfg, test=True)
    latents1, latents2 = latents.values()
    latents1_trafo = mapping.transform(latents1)
    visualize_results(cfg, labels, latents2, latents1_trafo)


if __name__ == '__main__':
    main()
