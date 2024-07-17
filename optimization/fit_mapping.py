"""Script for fitting a mapping between two latent spaces.

Specify the experiment parameters in `config/config_map.yaml`, then invoke with:
    python optimization/fit_mapping.py -m dataset=mnist,fmnist lamda=0,0.1 model1.seed=1,2,3 model2.seed=1,2,3 model1.name=pcktae model2.name=pcktae model1.latent_size=10,30,50 model2.latent_size=10,30,50 hydra.output_subdir=null

This defines a hydra multirun (sweep), which means the seed, name and latent_size parameters defined in the config file
will be overridden with all possible combinations of the passed parameters.
The base_dir parameter will be set at run time, please leave this as it is.
"""
import logging
import os.path
from pathlib import Path

import numpy as np
import torch
from omegaconf import DictConfig
import hydra
import hydra.core.global_hydra
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST, CIFAR10, FashionMNIST
from torchvision.transforms import transforms

from utils.model import load_model, get_transformations
from utils.sampler import sample_convex_hull, sample_uniformly, sample_convex_hulls_images, sample_equally_per_class_images
from utils.visualization import visualize_results

device = torch.device('cuda') if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'


# def get_train_latents(cfg):
#     in_channels, size = get_model_params(cfg.dataset)
#
#     model1 = load_model(cfg.model1.name, cfg.model1.path, in_channels, size, cfg.model1.latent_size)
#     model2 = load_model(cfg.model2.name, cfg.model2.path, in_channels, size, cfg.model2.latent_size)
#     dataset = get_dataset(cfg, cfg.model1.name)
#     dataloader = DataLoader(dataset, batch_size=len(dataset), shuffle=False)
#     train_data, val_data = sample_convex_hull(dataloader, model1, model2, cfg.num_samples) if cfg.sampling == "convex_hull" else sample_uniformly(dataloader, model1, model2, cfg.num_samples)
#
#     return train_data, val_data


def define_dataloader(file, file2, use_test_set=False):
    if file.strip("_")[0] != file2.strip("_")[0]:
        logging.error("The datasets are different")
    # Define the dataloaders
    name_dataset, name_model, size_of_the_latent, seed = file.strip(".pth").split("_")
    augumentation = get_transformations(name_model)
    if name_dataset.lower() == "mnist":
        from utils.dataloaders.full_dataloaders import DataLoaderMNIST
        dataloader = DataLoaderMNIST(transformation=augumentation, batch_size=64, seed=int(seed))
    if name_dataset.lower() == "fmnist":
        from utils.dataloaders.full_dataloaders import DataLoaderFashionMNIST
        dataloader = DataLoaderFashionMNIST(transformation=augumentation, batch_size=64, seed=int(seed))
    if name_dataset.lower() == "cifar10":
        from utils.dataloaders.full_dataloaders import DataLoaderCIFAR10
        dataloader = DataLoaderCIFAR10(transformation=augumentation, batch_size=64, seed=int(seed))
    if name_dataset.lower() == "cifar100":
        from utils.dataloaders.full_dataloaders import DataLoaderCIFAR100
        dataloader = DataLoaderCIFAR100(transformation=augumentation, batch_size=64, seed=int(seed))
    if use_test_set:
        full_dataset_images, full_dataset_labels = dataloader.get_full_test_dataset()
    else:
        full_dataset_images, full_dataset_labels = dataloader.get_full_train_dataset()
    return full_dataset_images, full_dataset_labels, len(np.unique(full_dataset_labels.numpy()))


def get_train_latents(cfg):
    in_channels, size = get_model_params(cfg.dataset)
    images, labels, n_classes = define_dataloader(f"{cfg.dataset}_{cfg.model1.name}_{cfg.model1.latent_size}_{cfg.model1.seed}.pth", f"{cfg.dataset}_{cfg.model2.name}_{cfg.model2.latent_size}_{cfg.model2.seed}.pth", use_test_set=False)
    images = images.type(torch.float32)
    labels = labels.type(torch.float32)
    model1 = load_model(cfg.model1.name, cfg.model1.path, in_channels, size, cfg.model1.latent_size)
    model2 = load_model(cfg.model2.name, cfg.model2.path, in_channels, size, cfg.model2.latent_size)

    if cfg.sampling == "convex_hull":
        (train_images, train_labels), (val_images, val_labels) = sample_convex_hulls_images(cfg.num_samples, images,
                                                                                             labels, model1, val_samples=1000)
    elif cfg.sampling == "equally":
        (train_images, train_labels), (val_images, val_labels) = sample_equally_per_class_images(cfg.num_samples, images,
                                                                                            labels, val_samples=1000)
    else:
        raise ValueError(f"Invalid sampling method: {cfg.sampling}")
    latents1 = model1.get_latent_space(train_images.to(device)).detach()
    latents2 = model2.get_latent_space(train_images.to(device)).detach()
    val_latents1 = model1.get_latent_space(val_images.to(device)).detach()
    val_latents2 = model2.get_latent_space(val_images.to(device)).detach()
    return (latents1, latents2, train_labels), (val_latents1, val_latents2, val_labels)


def get_test_latents(cfg):
    in_channels, size = get_model_params(cfg.dataset)

    model1 = load_model(cfg.model1.name, cfg.model1.path, in_channels, size, cfg.model1.latent_size)
    model2 = load_model(cfg.model2.name, cfg.model2.path, in_channels, size, cfg.model2.latent_size)
    dataset = get_dataset(cfg, cfg.model1.name, train=False)
    dataloader = DataLoader(dataset, batch_size=128, shuffle=False)
    latents1, labels = model1.get_latent_space_from_dataloader(dataloader)
    latents2, _ = model2.get_latent_space_from_dataloader(dataloader)
    return latents1, latents2, labels


def get_dataset(cfg, model_name, train=True):
    transformations = get_transformations(model_name)
    transformations = transforms.Compose(transformations)
    dataset_dict = {
        'mnist': MNIST,
        'fmnist': FashionMNIST,
        'cifar10': CIFAR10
    }
    dataset = dataset_dict[cfg.dataset](root=cfg.base_dir / 'data', train=train, transform=transformations, download=True)
    return dataset


def get_model_params(dataset):
    param_dict = {
        'mnist': (1, 7),
        'cifar10': (3, 8),
        'fmnist': (1, 7)
    }
    return param_dict[dataset.lower()]


def create_mapping(cfg, latents1, latents2, val_latents1=None, val_latents2=None, do_print=True):
    if cfg.mapping.lower() == 'linear':
        from optimization.optimizer import LinearFitting
        mapping = LinearFitting(latents1, latents2, lamda=cfg.lamda, do_print=do_print)
    elif cfg.mapping.lower() == 'affine':
        from optimization.optimizer import AffineFitting
        mapping = AffineFitting(latents1, latents2, lamda=cfg.lamda, do_print=do_print)
    elif cfg.mapping.lower() == 'neuralnetwork':
        from optimization.optimizer import NeuralNetworkFitting
        dropout = 0.3 if cfg.dropout else 0
        noise_sigma = 0.4 if cfg.noisy else 0
        mapping = NeuralNetworkFitting(latents1, latents2, hidden_dim=cfg.hidden_size, lamda=cfg.lamda, z1_val=val_latents1, z2_val=val_latents2, learning_rate=cfg.learning_rate, epochs=cfg.epochs, do_print=do_print, dropout=dropout, noise_sigma=noise_sigma)
    elif cfg.mapping.lower() == 'kernel':
        from optimization.optimizer import KernelFitting
        mapping = KernelFitting(latents1, latents2, lamda=cfg.lamda, gamma=cfg.gamma, do_print=do_print)
    elif cfg.mapping.lower() == 'adaptive':
        from optimization.optimizer import AdaptiveFitting, NeuralNetworkFitting
        mlp_mapping = NeuralNetworkFitting(latents1, latents2, hidden_dim=cfg.hidden_size, lamda=cfg.lamda, z1_val=val_latents1, z2_val=val_latents2, learning_rate=cfg.learning_rate, epochs=cfg.epochs, do_print=do_print)
        mapping = AdaptiveFitting(mlp_mapping, latents1, latents2, hidden_dim=cfg.hidden_size, lamda=cfg.lamda, z1_val=val_latents1, z2_val=val_latents2, learning_rate=cfg.learning_rate)
    elif cfg.mapping.lower() == 'hybrid':
        from optimization.optimizer import HybridFitting, LinearFitting
        path = cfg.base_dir / "results/transformations/mapping_files" / cfg.model1.name.upper() / f"{cfg.dataset.upper()}_{cfg.model1.name.upper()}_{cfg.model1.latent_size}_{cfg.model1.seed}>{cfg.dataset.upper()}_{cfg.model2.name.upper()}_{cfg.model2.latent_size}_{cfg.model2.seed}>Linear_{cfg.num_samples}_{cfg.lamda}_{cfg.sampling}"
        linear_mapping = LinearFitting.from_file(path)
        dropout = 0.3 if cfg.dropout else 0
        noise_sigma = 0.4 if cfg.noisy else 0
        mapping = HybridFitting(linear_mapping, latents1, latents2, hidden_dim=cfg.hidden_size, lamda=cfg.lamda, z1_val=val_latents1, z2_val=val_latents2, learning_rate=cfg.learning_rate, dropout=dropout, noise_sigma=noise_sigma)
    else:
        raise ValueError("Invalid experiment name")
    return mapping


def evaluate_mapping(cfg, mapping):
    latents1, latents2, labels = get_test_latents(cfg)
    latents1_trafo = mapping.transform(latents1)
    cfg.storage_path = cfg.base_dir / "results/transformations/figures" / cfg.model1.name.upper()
    visualize_results(cfg, labels, latents2, latents1_trafo)


def get_mapping_name(dataset, model1, model1_latent_size, model1_seed, model2, model2_latent_size, model2_seed, mapping, num_samples, lamda, sampling, dropout=None, noisy=False, hidden_size=None):
    if mapping in ["NeuralNetwork", "Hybrid"]:
        dropout = "dropout" if dropout else "noisy" if noisy else "nodropout"
        return f"{dataset.upper()}_{model1.upper()}_{model1_latent_size}_{model1_seed}>{dataset.upper()}_{model2.upper()}_{model2_latent_size}_{model2_seed}>{mapping}_{num_samples}_{lamda}_{dropout}_{hidden_size}_{sampling}"
    return f"{dataset.upper()}_{model1.upper()}_{model1_latent_size}_{model1_seed}>{dataset.upper()}_{model2.upper()}_{model2_latent_size}_{model2_seed}>{mapping}_{num_samples}_{lamda}_{sampling}"


def get_model_path(dataset, model, latent_size, seed):
    base_dir = Path(os.path.abspath(__file__)).parent.parent
    return base_dir / "models/checkpoints" / model.upper() / dataset.upper() /f"{dataset.upper()}_{model.upper()}_{latent_size}_{seed}.pth"


def get_mapping_params(filename):
    parts = filename.split("_")
    dataset = parts[0]
    model1 = parts[1]
    model1_latent_size = int(parts[2])
    model1_seed = int(parts[3])
    model2 = parts[4]
    model2_latent_size = int(parts[5])
    model2_seed = int(parts[6])
    mapping = parts[7]
    num_samples = int(parts[8])
    lamda = float(parts[9])
    sampling = parts[-1]
    if mapping in ["NeuralNetwork", "Hybrid"]:
        dropout = True if parts[10] == "dropout" else False
        noisy = True if parts[11] == "noisy" else False
        hidden_size = int(parts[11])
        return dataset, model1, model1_latent_size, model1_seed, model2, model2_latent_size, model2_seed, mapping, num_samples, lamda, sampling, dropout, noisy, hidden_size
    return dataset, model1, model1_latent_size, model1_seed, model2, model2_latent_size, model2_seed, mapping, num_samples, lamda, sampling


def get_params_from_model_name(filename):
    parts = filename.rstrip(".pth").split("_")
    dataset = parts[0]
    model_name = parts[1]
    latent_size = int(parts[2])
    seed = int(parts[3])
    return dataset, model_name, latent_size, seed


@hydra.main(version_base="1.1", config_path="../config", config_name="config_map")
def main(cfg : DictConfig) -> None:
    # check if models are equal
    if cfg.model1.name == cfg.model2.name and cfg.model1.seed == cfg.model2.seed:
        return
    if cfg.model1.latent_size != cfg.model2.latent_size:
        return
    cfg.base_dir = Path(hydra.utils.get_original_cwd()).parent
    cfg.model1.path = get_model_path(cfg.dataset, cfg.model1.name, cfg.model1.latent_size, cfg.model1.seed)
    cfg.model2.path = get_model_path(cfg.dataset, cfg.model2.name, cfg.model2.latent_size, cfg.model2.seed)
    storage_path = cfg.base_dir / "results/transformations/mapping_files" / cfg.model1.name.upper() / get_mapping_name(cfg.dataset, cfg.model1.name, cfg.model1.latent_size, cfg.model1.seed, cfg.model2.name, cfg.model2.latent_size, cfg.model2.seed, cfg.mapping, cfg.num_samples, cfg.lamda, cfg.sampling, cfg.dropout, cfg.noisy, cfg.hidden_size)
    if os.path.exists(storage_path.with_suffix(".pth")) or os.path.exists(storage_path.with_suffix(".npz")) or os.path.exists(storage_path.with_suffix(".npy")):
        print(f"Mapping already exists: {storage_path.name}")
        return

    (latents1, latents2, labels), (val_latents1, val_latents2, val_labels) = get_train_latents(cfg)

    if cfg.mapping in ["NeuralNetwork", "Adaptive", "Hybrid"]:
        mapping = create_mapping(cfg, latents1, latents2, val_latents1, val_latents2)
    else:
        mapping = create_mapping(cfg, latents1, latents2)
    mapping.fit()
    mapping.save_results(storage_path)

    evaluate_mapping(cfg, mapping)


if __name__ == '__main__':
    main()
