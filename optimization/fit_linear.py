from pathlib import Path
import torch
from omegaconf import DictConfig
import hydra

from stitching.stitching import get_transformations, load_model
from optimizer import AffineFitting, LinearFitting
from utils.sampler import simple_sampler

device = torch.device('cuda') if torch.cuda.is_available() else 'cpu'


def get_latents(cfg):
    """For both models, load the latent vectors if latent_path is provided, else load the models and sample the
        latent vectors."""
    torch.manual_seed(0)
    indices = torch.randperm(60000)[:cfg.num_samples]
    latents = {}
    for model_name in ['model1', 'model2']:
        if 'latents_path' in cfg[model_name]:
            labels = torch.load(cfg[model_name].label_path, map_location=device)
            z = torch.load(cfg[model_name].latents_path, map_location=device)
            latents[model_name] = z[indices]
        else:
            model = load_model(cfg[model_name].name, cfg[model_name].path)
            transforms = get_transformations(cfg[model_name].name)
            latents[model_name] = simple_sampler(indices, model, transforms, device, seed=cfg[model_name].seed)

    return latents


@hydra.main(config_path="../config", config_name="config_resnet")
def main(cfg : DictConfig) -> None:
    cfg.base_dir = Path(hydra.utils.get_original_cwd()).parent
    latents1, latents2 = get_latents(cfg).values()

    # Linear transform
    linear_fitting = LinearFitting(latents1, latents2, lamda=0.01)
    linear_fitting.solve_problem()
    storage_path = Path(cfg.storage_path) / f"Linear_{cfg.model1.name}_{cfg.model1.seed}_{cfg.model2.name}_{cfg.model2.seed}_{cfg.num_samples}"
    linear_fitting.save_results(storage_path)
    del linear_fitting

    # Affine transform
    affine_fitting = AffineFitting(latents1, latents2, lamda=0.01)
    affine_fitting.solve_problem()
    storage_path = Path(cfg.storage_path) / f"Affine_{cfg.model1.name}_{cfg.model1.seed}_{cfg.model2.name}_{cfg.model2.seed}_{cfg.num_samples}"
    affine_fitting.save_results(storage_path)


if __name__ == '__main__':
    main()
