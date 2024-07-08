"""Invoke with:
    python optimization/fit_mapping.py --config-name config_map -m dataset=mnist,fmnist lamda=0,0.001,0.01 model1.seed=1,2,3 model2.seed=1,2,3 model1.name=vae model2.name=vae model1.latent_size=10,30,50 model2.latent_size=10,30,50 hydra.output_subdir=null
"""

from pathlib import Path

import torch
from omegaconf import DictConfig
import hydra
import hydra.core.global_hydra
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST, CIFAR10, FashionMNIST
from torchvision.transforms import transforms

from utils.model import load_model, get_transformations
from utils.sampler import sample_convex_hull
from utils.visualization import visualize_results

device = torch.device('cuda') if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'
hydra.core.global_hydra.GlobalHydra.instance().clear()


def get_train_latents(cfg):
    train_latents = {}
    val_latents = {}
    in_channels, size = get_model_params(cfg.dataset)

    for model_name in ['model1', 'model2']:
        model = load_model(cfg[model_name].name, cfg[model_name].path, in_channels, size, cfg[model_name].latent_size)
        dataset = get_dataset(cfg, model_name)
        dataloader = DataLoader(dataset, batch_size=cfg.num_samples)
        _, train_latents[model_name], train_labels, _, val_latents[model_name], val_labels = sample_convex_hull(dataloader, model, cfg.num_samples)

    return train_latents, train_labels, val_latents, val_labels


def get_test_latents(cfg):
    latents = {}
    in_channels, size = get_model_params(cfg.dataset)

    for model_name in ['model1', 'model2']:
        model = load_model(cfg[model_name].name, cfg[model_name].path, in_channels, size, cfg[model_name].latent_size)
        dataset = get_dataset(cfg, model_name, train=False)
        dataloader = DataLoader(dataset, batch_size=128, shuffle=False)
        latents[model_name], labels = model.get_latent_space_from_dataloader(dataloader)
    return latents, labels


def get_dataset(cfg, model_name, train=True):
    transformations = get_transformations(cfg[model_name].name)
    transformations = transforms.Compose(transformations)
    if cfg.dataset == 'mnist':
        dataset = MNIST(root=cfg.base_dir / 'data', train=train, transform=transformations, download=True)
    elif cfg.dataset == 'fmnist':
        dataset = FashionMNIST(root=cfg.base_dir / 'data', train=train, transform=transformations, download=True)
    elif cfg.dataset == 'cifar10':
        dataset = CIFAR10(root=cfg.base_dir / 'data', train=train, transform=transformations, download=True)
    else:
        raise ValueError("Invalid dataset")
    return dataset


def get_model_params(dataset):
    if dataset == 'mnist':
        in_channels = 1
        size = 7
    elif dataset == 'cifar10':
        in_channels = 3
        size = 8
    elif dataset == 'fmnist':
        in_channels = 1
        size = 7
    else:
        raise ValueError("Invalid dataset")
    return in_channels, size


def create_mapping(cfg, latents1, latents2, val_latents1=None, val_latents2=None, do_print=True):
    if cfg.mapping.lower() == 'linear':
        from optimization.optimizer import LinearFitting
        mapping = LinearFitting(latents1, latents2, lamda=cfg.lamda, do_print=do_print)
    elif cfg.mapping.lower() == 'affine':
        from optimization.optimizer import AffineFitting
        mapping = AffineFitting(latents1, latents2, lamda=cfg.lamda, do_print=do_print)
    elif cfg.mapping.lower() == 'neuralnetwork':
        from optimization.optimizer import NeuralNetworkFitting
        mapping = NeuralNetworkFitting(latents1, latents2, hidden_dim=cfg.hidden_size, lamda=cfg.lamda, z1_val=val_latents1, z2_val=val_latents2, learning_rate=cfg.learning_rate, epochs=cfg.epochs, do_print=do_print)
    elif cfg.mapping.lower() == 'kernel':
        from optimization.optimizer import KernelFitting
        mapping = KernelFitting(latents1, latents2, lamda=cfg.lamda, gamma=cfg.gamma, do_print=do_print)
    elif cfg.mapping.lower() == 'adaptive':
        from optimization.optimizer import AdaptiveFitting
        mapping = AdaptiveFitting(latents1, latents2, hidden_dim=cfg.hidden_size, lamda=cfg.lamda, z1_val=val_latents1, z2_val=val_latents2, learning_rate=cfg.learning_rate)
    else:
        raise ValueError("Invalid experiment name")
    return mapping


def evaluate_mapping(cfg, mapping):
    latents, labels = get_test_latents(cfg)
    latents1, latents2 = latents.values()
    latents1_trafo = mapping.transform(latents1)
    cfg.storage_path = cfg.base_dir / "results/transformations/figures" / cfg.model1.name.upper()
    visualize_results(cfg, labels, latents2, latents1_trafo)


@hydra.main(version_base="1.1", config_path="../config", config_name="config_map")
def main(cfg : DictConfig) -> None:
    # check if models are equal
    if cfg.model1.name == cfg.model2.name and cfg.model1.seed == cfg.model2.seed:
        return
    cfg.base_dir = Path(hydra.utils.get_original_cwd()).parent
    cfg.model1.path = cfg.base_dir / "models/checkpoints" / f"{cfg.model1.name.upper()}/{cfg.dataset.upper()}/{cfg.dataset.upper()}_{cfg.model1.name.upper()}_{cfg.model1.latent_size}_{cfg.model1.seed}.pth"
    cfg.model2.path = cfg.base_dir / "models/checkpoints" / f"{cfg.model2.name.upper()}/{cfg.dataset.upper()}/{cfg.dataset.upper()}_{cfg.model2.name.upper()}_{cfg.model2.latent_size}_{cfg.model2.seed}.pth"
    latents, labels, val_latents, val_labels = get_train_latents(cfg)
    latents1, latents2 = latents.values()
    val_latents1, val_latents2 = val_latents.values()
    del latents, val_latents

    storage_path = cfg.base_dir / "results/transformations/mapping_files" / cfg.model1.name.upper() / f"{cfg.dataset.upper()}_{cfg.model1.name.upper()}_{cfg.model1.latent_size}_{cfg.model1.seed}>{cfg.dataset.upper()}_{cfg.model2.name.upper()}_{cfg.model2.latent_size}_{cfg.model2.seed}>{cfg.mapping}_{cfg.num_samples}_{cfg.lamda}_{cfg.sampling}"
    # if storage_path.exists():
    #     return
    if cfg.mapping in ["NeuralNetwork", "Adaptive"]:
        mapping = create_mapping(cfg, latents1, latents2, val_latents1, val_latents2)
    else:
        mapping = create_mapping(cfg, latents1, latents2)
    if cfg.mapping == "Adaptive":
        mapping.mlp_model = torch.load(str(storage_path).replace("Adaptive", "NeuralNetwork") + ".pth")
    mapping.fit()
    mapping.save_results(storage_path)

    evaluate_mapping(cfg, mapping)


if __name__ == '__main__':
    main()
