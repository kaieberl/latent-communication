"""Invoke with:
    python optimization/fit_mapping.py --config-name config_map -m dataset=mnist,fmnist lamda=0,0.001,0.01 model1.seed=1,2,3 model2.seed=1,2,3 model1.name=vae model2.name=vae model1.latent_size=10,30,50 model2.latent_size=10,30,50 hydra.output_subdir=null
"""

from pathlib import Path

import torch
from omegaconf import DictConfig
import hydra
import hydra.core.global_hydra
from torch.utils.data import DataLoader, SubsetRandomSampler
from torchvision.datasets import MNIST, CIFAR10, FashionMNIST
from torchvision.transforms import transforms

from utils.model import load_model, get_transformations
from utils.visualization import visualize_results

device = torch.device('cuda') if torch.cuda.is_available() else 'cpu'
hydra.core.global_hydra.GlobalHydra.instance().clear()


def get_latents(cfg, val=False, test=False):
    """For both models, load the latent vectors if latent_path is provided, else load the models and sample the
        latent vectors.

    Args:
        cfg (DictConfig): Configuration dictionary
        val (bool): If True, load the validation latent vectors
        test (bool): If True, load the test latent vectors

    Returns:
        dict: Dictionary containing the latent vectors for both models
        torch.Tensor: Labels
    """
    latents = {}
    if cfg.dataset == 'mnist':
        in_channels = 1
        size = 7
    elif cfg.dataset == 'cifar10':
        in_channels = 3
        size = 8
    elif cfg.dataset == 'fmnist':
        in_channels = 1
        size = 7
    else:
        raise ValueError("Invalid dataset")

    if not test:
        torch.manual_seed(0)
        for model_name in ['model1', 'model2']:
            if 'train_latents_path' in cfg[model_name]:
                labels = torch.load(cfg.train_label_path, map_location=device)
                z = torch.load(cfg[model_name].train_latents_path, map_location=device)
                indices = []
                for i in range(10):
                    indices += torch.where(torch.load(cfg.train_label_path, map_location=device) == i)[0][
                               :cfg.num_samples // 10].tolist()
                latents[model_name] = z[indices]
            else:
                model = load_model(cfg[model_name].name, cfg[model_name].path, in_channels, size, cfg[model_name].latent_size)
                transformations = get_transformations(cfg[model_name].name)
                transformations = transforms.Compose(transformations)
                if cfg.dataset == 'mnist':
                    dataset = MNIST(root=cfg.base_dir / 'data', train=True, transform=transformations, download=True)
                elif cfg.dataset == 'fmnist':
                    dataset = FashionMNIST(root=cfg.base_dir / 'data', train=True, transform=transformations, download=True)
                elif cfg.dataset == 'cifar10':
                    dataset = CIFAR10(root=cfg.base_dir / 'data', train=True, transform=transformations, download=True)
                else:
                    raise ValueError("Invalid dataset")
                indices = []
                if not val:
                    for i in range(10):
                        indices += torch.where(torch.tensor(dataset.targets) == i)[0][:cfg.num_samples // 10].tolist()
                else:
                    for i in range(10):
                        indices += torch.where(torch.tensor(dataset.targets) == i)[0][cfg.num_samples // 10:(cfg.num_samples + 5000) // 10].tolist()
                dataloader = DataLoader(dataset, batch_size=cfg.num_samples, sampler=SubsetRandomSampler(indices))
                latents[model_name], labels = model.get_latent_space_from_dataloader(dataloader)
    else:
        for model_name in ['model1', 'model2']:
            if 'test_latents_path' in cfg[model_name]:
                labels = torch.load(cfg.test_label_path, map_location=device)
                z = torch.load(cfg[model_name].test_latents_path, map_location=device)
                latents[model_name] = z
            else:
                model = load_model(cfg[model_name].name, cfg[model_name].path, in_channels, size, cfg[model_name].latent_size)
                transformations = get_transformations(cfg[model_name].name)
                transformations = transforms.Compose(transformations)
                if cfg.dataset == 'mnist':
                    dataset = MNIST(root=cfg.base_dir / 'data', train=False, transform=transformations, download=True)
                elif cfg.dataset == 'fmnist':
                    dataset = FashionMNIST(root=cfg.base_dir / 'data', train=False, transform=transformations, download=True)
                elif cfg.dataset == 'cifar10':
                    dataset = CIFAR10(root=cfg.base_dir / 'data', train=False, transform=transformations, download=True)
                else:
                    raise ValueError("Invalid dataset")
                dataloader = DataLoader(dataset, batch_size=128, shuffle=False)
                latents[model_name], labels = model.get_latent_space_from_dataloader(dataloader)
    return latents, labels


def create_mapping(cfg, latents1, latents2, val_latents1=None, val_latents2=None, do_print=True):
    if cfg.mapping == 'Linear':
        from optimization.optimizer import LinearFitting
        mapping = LinearFitting(latents1, latents2, lamda=cfg.lamda, do_print=do_print)
    elif cfg.mapping == 'Affine':
        from optimization.optimizer import AffineFitting
        mapping = AffineFitting(latents1, latents2, lamda=cfg.lamda, do_print=do_print)
    elif cfg.mapping == 'NeuralNetwork':
        from optimization.optimizer import NeuralNetworkFitting
        mapping = NeuralNetworkFitting(latents1, latents2, hidden_dim=cfg.hidden_size, lamda=cfg.lamda, learning_rate=cfg.learning_rate, epochs=cfg.epochs, do_print=do_print)
    elif cfg.mapping == 'Kernel':
        from optimization.optimizer import KernelFitting
        mapping = KernelFitting(latents1, latents2, lamda=cfg.lamda, gamma=cfg.gamma, do_print=do_print)
    elif cfg.mapping == 'Decouple':
        from optimization.optimizer import DecoupleFitting
        mapping = DecoupleFitting(latents1, latents2, lamda=cfg.lamda, do_print=do_print)
    elif cfg.mapping == 'DecoupleAffine':
        from optimization.optimizer import DecoupleFitting
        mapping = DecoupleFitting(latents1, latents2, lamda=cfg.lamda, do_print=do_print,mapping_type='Affine')
    else:
        raise ValueError("Invalid experiment name")
    return mapping


@hydra.main(version_base="1.1", config_path="../config")
def main(cfg : DictConfig) -> None:
    # check if models are equal
    if cfg.model1.name == cfg.model2.name and cfg.model1.seed == cfg.model2.seed:
        return
    cfg.base_dir = Path(hydra.utils.get_original_cwd()).parent
    cfg.model1.path = cfg.base_dir / "models/checkpoints" / f"{cfg.model1.name.upper()}/{cfg.dataset.upper()}/{cfg.dataset.upper()}_{cfg.model1.name.upper()}_{cfg.model1.latent_size}_{cfg.model1.seed}.pth"
    cfg.model2.path = cfg.base_dir / "models/checkpoints" / f"{cfg.model2.name.upper()}/{cfg.dataset.upper()}/{cfg.dataset.upper()}_{cfg.model2.name.upper()}_{cfg.model2.latent_size}_{cfg.model2.seed}.pth"
    latents1, latents2 = get_latents(cfg)[0].values()
    # val_latents1, val_latents2 = get_latents(cfg, val=True)[0].values()

    storage_path = cfg.base_dir / "results/transformations/mapping_files" / cfg.model1.name.upper() / f"{cfg.dataset.upper()}_{cfg.model1.name.upper()}_{cfg.model1.latent_size}_{cfg.model1.seed}>{cfg.dataset.upper()}_{cfg.model2.name.upper()}_{cfg.model2.latent_size}_{cfg.model2.seed}>{cfg.mapping}_{cfg.num_samples}_{cfg.lamda}_equally"
    if storage_path.exists():
        return
    mapping = create_mapping(cfg, latents1, latents2)
    mapping.fit()
    mapping.save_results(storage_path)

    latents, labels = get_latents(cfg, test=True)
    latents1, latents2 = latents.values()
    latents1_trafo = mapping.transform(latents1)
    cfg.storage_path = cfg.base_dir / "results/transformations/figures" / cfg.model1.name.upper()
    visualize_results(cfg, labels, latents2, latents1_trafo)


if __name__ == '__main__':
    main()
