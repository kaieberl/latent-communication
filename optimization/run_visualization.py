"""Creates a PCA scatter plot of the latent space and associated mapping error.

First fit a mapping using `fit_mapping.py`, then invoke with:
    python run_visualization.py
"""

import numpy as np
from omegaconf import DictConfig
import hydra
import torch
from pathlib import Path

from sklearn.manifold import TSNE
from torch.utils.data import DataLoader

from optimization.fit_mapping import get_dataset
from utils.visualization import visualize_mapping_error, visualize_latent_space
from utils.model import load_models
from stitching.stitching import load_mapping

device = torch.device('cuda') if torch.cuda.is_available() else 'cpu'


def get_latent_space_from_dataloader(model, dataloader, max_nr_batches=10):
    print("Calculating latent_spaces")
    latents = []
    labels = []
    N = 0
    for data in dataloader:
        N += 1
        if N < max_nr_batches:
            image, label = data
            image, label = image.to(device), label.to(device)
            latent = model.get_latent_space(image)
            latents.append(latent)
            labels.append(label)
        else:
            break
    print("Finished calculating latent_spaces")
    return torch.cat(latents), torch.cat(labels)


@hydra.main(version_base="1.1", config_path="../config", config_name="config_map")
def main(cfg: DictConfig):
    cfg.base_dir = Path(hydra.utils.get_original_cwd()).parent
    dataset = get_dataset(cfg, model_name=cfg.model1.name, train=False)
    test_loader = DataLoader(dataset, batch_size=128, shuffle=False)

    # Load model
    cfg.model1.path = cfg.base_dir / "models/checkpoints" / f"{cfg.model1.name.upper()}/{cfg.dataset.upper()}/{cfg.dataset.upper()}_{cfg.model1.name.upper()}_{cfg.model1.latent_size}_{cfg.model1.seed}.pth"
    cfg.model2.path = cfg.base_dir / "models/checkpoints" / f"{cfg.model2.name.upper()}/{cfg.dataset.upper()}/{cfg.dataset.upper()}_{cfg.model2.name.upper()}_{cfg.model2.latent_size}_{cfg.model2.seed}.pth"
    model1, model2 = load_models(cfg)
    model1 = model1.to(device)
    model2 = model2.to(device)
    model1.eval()
    model2.eval()

    # Load mapping
    mapping = load_mapping(cfg)

    # Get the latent space of the test set
    latents1, labels = get_latent_space_from_dataloader(model1, test_loader, max_nr_batches=1000)
    latents2, labels2 = get_latent_space_from_dataloader(model2, test_loader, max_nr_batches=1000)

    latents1 = latents1.detach().numpy()
    latents2 = latents2.detach().numpy()
    labels = labels.detach().numpy()

    # Get transformed latent space
    latents1_trafo = mapping.transform(latents1).numpy()

    # Plot the results
    print(f"Mean error for {cfg.mapping} mapping: {np.mean(np.linalg.norm(latents2 - latents1_trafo, axis=1)):.4f}")
    cfg.storage_path = cfg.base_dir / "results/transformations/figures" / cfg.model1.name.upper()
    if not cfg.storage_path.exists():
        cfg.storage_path.mkdir(parents=True)
    visualize_latent_space(latents1, labels,
                           cfg.storage_path / f"latent_space_{cfg.visualization}_{cfg.model1.name}_{cfg.model1.seed}_test.png", mode=cfg.visualization)
    pca, _ = visualize_latent_space(latents2, labels,
                                    cfg.storage_path / f"latent_space_{cfg.visualization}_{cfg.model2.name}_{cfg.model2.seed}_test.png", mode=cfg.visualization)
    visualize_latent_space(latents1_trafo, labels,
                           cfg.storage_path / f"latent_space_{cfg.visualization}_{cfg.mapping}_{cfg.model1.name}_{cfg.model1.latent_size}_{cfg.model1.seed}_{cfg.model1.name}_{cfg.model2.latent_size}_{cfg.model2.seed}_test_{cfg.mapping}_{cfg.num_samples}.png",
                           pca=pca, mode=cfg.visualization)
    errors = np.linalg.norm(latents2 - latents1_trafo, axis=1)
    if cfg.visualization == 'pca':
        latents2_2d = pca.transform(latents1_trafo)
    elif cfg.visualization == 'tsne':
        tsne = TSNE(n_components=2, random_state=0)
        latents2_2d = tsne.fit_transform(latents1_trafo)
    else:
        raise ValueError(f"Invalid visualization method: {cfg.visualization}")
    visualize_mapping_error(latents2_2d, errors,
                            cfg.storage_path / f"mapping_error_{cfg.model1.name}_{cfg.model1.latent_size}_{cfg.model1.seed}_{cfg.model1.name}_{cfg.model2.latent_size}_{cfg.model2.seed}_test_{cfg.mapping}_{cfg.num_samples}.png")


if __name__ == "__main__":
    main()
