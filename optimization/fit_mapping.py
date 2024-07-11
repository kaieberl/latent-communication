"""Invoke with:
    python optimization/fit_mapping.py --config-name config_map -m dataset=mnist,fmnist lamda=0,0.001,0.01 model1.seed=1,2,3 model2.seed=1,2,3 model1.name=vae model2.name=vae model1.latent_size=10,30,50 model2.latent_size=10,30,50 hydra.output_subdir=null
"""
import os.path
from pathlib import Path

import torch
from omegaconf import DictConfig
import hydra
import hydra.core.global_hydra
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST, CIFAR10, FashionMNIST
from torchvision.transforms import transforms

from utils.model import load_model, get_transformations
from utils.sampler import sample_convex_hull, sample_uniformly
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
        _, train_latents[model_name], train_labels, _, val_latents[model_name], val_labels = sample_convex_hull(dataloader, model, cfg.num_samples) if cfg.sampling == "convex_hull" else sample_uniformly(dataloader, model, cfg.num_samples)

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
        from optimization.optimizer import AdaptiveFitting
        mapping = AdaptiveFitting(latents1, latents2, hidden_dim=cfg.hidden_size, lamda=cfg.lamda, z1_val=val_latents1, z2_val=val_latents2, learning_rate=cfg.learning_rate)
    elif cfg.mapping.lower() == 'hybrid':
        from optimization.optimizer import HybridFitting, AffineFitting
        path = cfg.base_dir / "results/transformations/mapping_files" / cfg.model1.name.upper() / f"{cfg.dataset.upper()}_{cfg.model1.name.upper()}_{cfg.model1.latent_size}_{cfg.model1.seed}>{cfg.dataset.upper()}_{cfg.model2.name.upper()}_{cfg.model2.latent_size}_{cfg.model2.seed}>Affine_{cfg.num_samples}_{cfg.lamda}_{cfg.sampling}"
        affine_mapping = AffineFitting.from_file(path)
        dropout = 0.3 if cfg.dropout else 0
        noise_sigma = 0.4 if cfg.noisy else 0
        mapping = HybridFitting(affine_mapping, latents1, latents2, hidden_dim=cfg.hidden_size, lamda=cfg.lamda, z1_val=val_latents1, z2_val=val_latents2, learning_rate=cfg.learning_rate, dropout=dropout, noise_sigma=noise_sigma)
    else:
        raise ValueError("Invalid experiment name")
    return mapping


def evaluate_mapping(cfg, mapping):
    latents, labels = get_test_latents(cfg)
    latents1, latents2 = latents.values()
    latents1_trafo = mapping.transform(latents1)
    # if os.path.exists(cfg.base_dir / "results/transformations/mapping_files/results.csv"):
    #     results = pd.read_csv(cfg.base_dir / "results/transformations/mapping_files/results.csv")
    # else:
    #     results = pd.DataFrame()
    # model1, model2 = load_models(cfg)
    # images = get_dataset(cfg, 'model1', train=False)
    # criterion = torch.nn.MSELoss()
    # mse_loss, mse_loss_model1, mse_loss_model2 = compute_losses(model1, model2, mapping, images, criterion)
    # results = results.append({
    #     "dataset": cfg.dataset,
    #     "model1": cfg.model1.name,
    #     "model2": cfg.model2.name,
    #     "mapping": cfg.mapping,
    #     "lambda": cfg.lamda,
    #     "num_samples": cfg.num_samples,
    #     "MSE_loss": mse_loss,
    #     "latent_dim": cfg.model1.latent_size,
    #     "MSE_loss_model1": mse_loss_model1,
    #     "MSE_loss_model2": mse_loss_model2,
    # }, ignore_index=True)
    # results.to_csv(cfg.base_dir / "results/transformations/mapping_files/results.csv")
    cfg.storage_path = cfg.base_dir / "results/transformations/figures" / cfg.model1.name.upper()
    visualize_results(cfg, labels, latents2, latents1_trafo)


def get_mapping_name(dataset, model1, model1_latent_size, model1_seed, model2, model2_latent_size, model2_seed, mapping, num_samples, lamda, sampling, dropout=None, noisy=False, hidden_size=None):
    if mapping in ["NeuralNetwork", "Hybrid"]:
        dropout = "dropout" if dropout else "noisy" if noisy else "nodropout"
        return f"{dataset.upper()}_{model1.upper()}_{model1_latent_size}_{model1_seed}>{dataset.upper()}_{model2.upper()}_{model2_latent_size}_{model2_seed}>{mapping}_{num_samples}_{lamda}_{dropout}_{hidden_size}_{sampling}"
    return f"{dataset.upper()}_{model1.upper()}_{model1_latent_size}_{model1_seed}>{dataset.upper()}_{model2.upper()}_{model2_latent_size}_{model2_seed}>{mapping}_{num_samples}_{lamda}_{sampling}"


def get_model_path(dataset, model, latent_size, seed):
    base_dir = Path(os.path.abspath(__file__)).parent.parent
    return base_dir / "models/checkpoints" / f"{model.upper()}/{dataset.upper()}/{dataset.upper()}_{model.upper()}_{latent_size}_{seed}.pth"


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
        return

    latents, labels, val_latents, val_labels = get_train_latents(cfg)
    latents1, latents2 = latents.values()
    val_latents1, val_latents2 = val_latents.values()
    del latents, val_latents

    if cfg.mapping in ["NeuralNetwork", "Adaptive", "Hybrid"]:
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
