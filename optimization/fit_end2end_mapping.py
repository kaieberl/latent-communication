from pathlib import Path

import torch
from omegaconf import DictConfig
import hydra
import hydra.core.global_hydra
from torch.utils.data import DataLoader

from optimization.fit_mapping import get_dataset, get_model_params, evaluate_mapping
from utils.model import load_model
from utils.sampler import sample_convex_hull

device = 'mps' if torch.backends.mps.is_available() else 'cpu'
hydra.core.global_hydra.GlobalHydra.instance().clear()


def get_train_loader(cfg):
    in_channels, size = get_model_params(cfg.dataset)

    for model_name in ['model1', 'model2']:
        model = load_model(cfg[model_name].name, cfg[model_name].path, in_channels, size, cfg[model_name].latent_size)
        dataset = get_dataset(cfg, model_name)
        dataloader = DataLoader(dataset, batch_size=cfg.num_samples)
        train_idxs, _, _, val_idxs, _, _ = sample_convex_hull(dataloader, model, cfg.num_samples)

    train_loader = DataLoader(dataset, batch_size=cfg.num_samples, sampler=train_idxs)
    val_loader = DataLoader(dataset, batch_size=cfg.num_samples, sampler=val_idxs)
    return train_loader, val_loader


def create_end2end_mapping(cfg, train_loader, val_loader=None, do_print=True):
    if cfg.mapping == 'NeuralNetwork':
        from optimization.end2end_optimizer import NeuralNetworkFitting
        mapping = NeuralNetworkFitting(train_loader, hidden_dim=cfg.hidden_size, lamda=cfg.lamda, val_loader=val_loader, learning_rate=cfg.learning_rate, epochs=cfg.epochs)
    else:
        raise ValueError(f"Invalid experiment name: {cfg.mapping}")
    return mapping


@hydra.main(version_base="1.1", config_path="../config", config_name="config_map")
def main(cfg : DictConfig) -> None:
    # check if models are equal
    if cfg.model1.name == cfg.model2.name and cfg.model1.seed == cfg.model2.seed:
        return
    cfg.base_dir = Path(hydra.utils.get_original_cwd()).parent
    cfg.model1.path = cfg.base_dir / "models/checkpoints" / f"{cfg.model1.name.upper()}/{cfg.dataset.upper()}/{cfg.dataset.upper()}_{cfg.model1.name.upper()}_{cfg.model1.latent_size}_{cfg.model1.seed}.pth"
    cfg.model2.path = cfg.base_dir / "models/checkpoints" / f"{cfg.model2.name.upper()}/{cfg.dataset.upper()}/{cfg.dataset.upper()}_{cfg.model2.name.upper()}_{cfg.model2.latent_size}_{cfg.model2.seed}.pth"
    train_loader, val_loader = get_train_loader(cfg)

    storage_path = cfg.base_dir / "results/transformations/mapping_files" / cfg.model1.name.upper() / f"{cfg.dataset.upper()}_{cfg.model1.name.upper()}_{cfg.model1.latent_size}_{cfg.model1.seed}>{cfg.dataset.upper()}_{cfg.model2.name.upper()}_{cfg.model2.latent_size}_{cfg.model2.seed}>{cfg.mapping}_{cfg.num_samples}_{cfg.lamda}_{cfg.sampling}"
    # if storage_path.exists():
    #     return
    mapping = create_end2end_mapping(cfg, train_loader, val_loader)
    if cfg.mapping == "Adaptive":
        mapping.mlp_model = torch.load(str(storage_path).replace("Adaptive", "NeuralNetwork") + ".pth")
    mapping.fit()
    mapping.save_results(storage_path)

    evaluate_mapping(cfg, mapping)


if __name__ == '__main__':
    main()
