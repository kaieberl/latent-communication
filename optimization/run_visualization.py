import numpy as np
from omegaconf import DictConfig
import hydra
import torch
from pathlib import Path

from utils.visualization import visualize_mapping_error, visualize_latent_space_pca
from utils.dataloaders.dataloader_mnist_single import DataLoaderMNIST
from utils.model import load_models, get_transformations
from stitching.stitching import load_mapping

device = torch.device('cuda') if torch.cuda.is_available() else 'cpu'
device='cpu'

def get_latent_space_from_dataloader(model, dataloader,max_nr_batches=10):
    print("Calculating latent_spaces")
    latents = []
    labels = []
    N=0
    for data in dataloader:
        N+=1
        if N<max_nr_batches:
            image, label = data
            image, label = image.to(device), label.to(device)
            latent = model.get_latent_space(image)
            latents.append(latent)
            labels.append(label)
        else:
            #skip for loop
            break
    print("Finished calculating latent_spaces")
    return torch.cat(latents), torch.cat(labels)

@hydra.main(version_base="1.1", config_path="../config")
def main(cfg : DictConfig) -> None:
    cfg.base_dir = Path(hydra.utils.get_original_cwd()).parent
    #Load test data set to plot
    data_loader_model = DataLoaderMNIST(128, get_transformations(cfg.model1.name), seed=10, base_path = cfg.base_dir)
    test_loader = data_loader_model.get_test_loader()
    print("loaded data")

    #Load model
    model1, model2 = load_models(cfg)
    model1 = model1.to(device)
    model2 = model2.to(device)
    model1.eval()
    model2.eval()

    #Load mapping
    mapping = load_mapping(cfg)

    #Get the latent space of the test set
    latents1,labels = get_latent_space_from_dataloader(model1, test_loader)
    latents2,labels2 = get_latent_space_from_dataloader(model2, test_loader)

    latents1 = latents1.detach().numpy()
    latents2 = latents2.detach().numpy()
    labels = labels.detach().numpy()

    #Get transformed latent space
    latents1_trafo = mapping.transform(latents1).detach().numpy()

    #Plot the results
    print("Plotting")
    print(f"Mean error for {cfg.mapping} mapping: {np.mean(np.linalg.norm(latents2 - latents1_trafo, axis=1)):.4f}")
    visualize_latent_space_pca(latents1, labels, f"../../figures/latent_space_pca_{cfg.model1.name}_{cfg.model1.seed}_test_.png")
    pca, _ = visualize_latent_space_pca(latents2, labels, f"../../figures/latent_space_pca_{cfg.model2.name}_{cfg.model2.seed}_test_.png")
    visualize_latent_space_pca(latents1_trafo, labels, f"../../figures/latent_space_pca_{cfg.mapping}_{cfg.model1.name}_seed{cfg.model1.seed}_{cfg.model1.name}_seed{cfg.model2.seed}_test_{cfg.mapping}_.png", pca=pca)
    #errors = np.linalg.norm(latents2 - latents1_trafo, axis=1)
    #visualize_mapping_error(latents2_2d, errors, f"../../figures/mapping_error_{cfg.model1.name}_seed{cfg.model1.seed}_{cfg.model1.name}_seed{cfg.model2.seed}_test_linear_.png")

if __name__ == "__main__":
    main()