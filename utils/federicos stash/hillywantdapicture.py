# %%
from pathlib import Path
import os

os.chdir("/Users/federicoferoggio/Documents/vs_code/latent-communication")

import itertools

import torch
import numpy as np
from tqdm import tqdm
from torchvision.datasets import MNIST
from torchvision.transforms import transforms
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

from models.definitions.PocketAutoencoder import PocketAutoencoder
from utils.dataloaders.dataloader_mnist_single import DataLoaderMNIST
from utils.visualization import (
    visualize_mapping_error,
    visualize_latent_space_pca,
    plot_latent_space,
    highlight_cluster,
)
from utils.sampler import *

DEVICE = torch.device("cuda") if torch.cuda.is_available() else "cpu"
augmentations = [transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))]


class Config:
    def __init__(self, **entries):
        self.__dict__.update(entries)

# %%
def create_mapping(cfg, latents1, latents2):
    if cfg.mapping == "Linear":
        from optimization.optimizer import LinearFitting

        mapping = LinearFitting(latents1, latents2, lamda=cfg.lamda)
    elif cfg.mapping == "Affine":
        from optimization.optimizer import AffineFitting

        mapping = AffineFitting(latents1, latents2, lamda=cfg.lamda)
    elif cfg.mapping == "NeuralNetwork":
        from optimization.optimizer import NeuralNetworkFitting

        mapping = NeuralNetworkFitting(
            latents1,
            latents2,
            hidden_dim=cfg.hidden_size,
            lamda=cfg.lamda,
            learning_rate=cfg.learning_rate,
            epochs=cfg.epochs,
        )
    else:
        raise ValueError("Invalid experiment name")
    return mapping


def visualize_modified_latent_space_pca(
    latents_trans,
    latents_2,
    labels,
    fig_path=None,
    anchors=None,
    pca=None,
    size=10,
    bg_alpha=1,
    alpha=1,
    title="2D PCA of Latent Space",
):
    """
    Visualizes the 2D latent space obtained from PCA.

    Args:
        latents_trans: A tensor of shape (N, dim) representing the first set of latent points.
        latents_2: A tensor of shape (N, dim) representing the second set of latent points.
        labels: A tensor of shape (N,) representing the labels for each latent point.
        fig_path: Optional; Path to save the figure.
        anchors: Optional; A tensor of shape (M, dim) representing anchor points in the latent space.
        pca: Optional; A PCA object to use for transforming the latent space.
        size: Optional; Size of the points in the plot.
        bg_alpha: Optional; Alpha value for the background points.
        alpha: Optional; Alpha value for the highlighted points.
    """
    # Convert lists to tensors if needed
    if isinstance(latents_trans, list):
        latents_trans = torch.tensor(latents_trans)
    if isinstance(latents_2, list):
        latents_2 = torch.tensor(latents_2)
    labels = np.asarray(labels)

    # Concatenate latent spaces
    latents = torch.cat([latents_trans, latents_2], dim=0)
    print(latents.shape)

    if pca is None:
        pca = PCA(n_components=2)
        latents_2d = pca.fit_transform(latents)
    else:
        latents_2d = pca.transform(latents)

    # Normalize latents
    minimum = latents_2d.min(axis=0)
    maximum = latents_2d.max(axis=0)
    latents_2d -= minimum
    latents_2d /= maximum

    # Separate the two datasets
    latents_trans_2d = latents_2d[: len(latents_trans)]
    latents_2_2d = latents_2d[len(latents_trans) :]

    # Create DataFrames for easy plotting
    latent_df_trans = pd.DataFrame(latents_trans_2d, columns=["x", "y"])
    latent_df_trans["target"] = labels

    latent_df_2 = pd.DataFrame(latents_2_2d, columns=["x", "y"])
    latent_df_2["target"] = labels

    # Plot the 2D latent space
    fig, ax = plt.subplots(figsize=(6, 6))
    cmap = plt.get_cmap("tab10")
    norm = plt.Normalize(
        latent_df_trans["target"].min(), latent_df_trans["target"].max()
    )

    ax = plot_latent_space(
        ax,
        latent_df_trans,
        targets=np.unique(labels),
        size=size,
        cmap=cmap,
        norm=norm,
        bg_alpha=bg_alpha,
        alpha=alpha,
        marker="2",
    )
    ax = plot_latent_space(
        ax,
        latent_df_2,
        targets=np.unique(labels),
        size=size,
        cmap=cmap,
        norm=norm,
        bg_alpha=bg_alpha,
        alpha=alpha,
        marker="1",
    )

    if anchors is not None:
        # plot anchors with star marker
        anchors_2d = pca.transform(anchors.cpu().detach().numpy())
        anchors_2d -= minimum
        anchors_2d /= maximum
        ax.scatter(anchors_2d[:, 0], anchors_2d[:, 1], marker="*", s=50, c="black")

    plt.title(title)
    if fig_path is not None:
        plt.savefig(fig_path)
    plt.show()

# %%
# Initialize models
model1 = PocketAutoencoder()
model2 = PocketAutoencoder()

# Load data
data_loader = DataLoaderMNIST(64, transformation=augmentations)
images, labels = data_loader.get_full_test_dataset()

# Generate combinations of parameters
num_samples_list = [100]
mapping_list = ["Linear", "Affine"]
lamda_list = [0.001]
combinations = list(itertools.product(num_samples_list, mapping_list, lamda_list))

# Change working directory
os.chdir("/Users/federicoferoggio/Documents/vs_code/latent-communication")

# Loop through combinations
for num_samples, mapping, lamda in combinations:
    parameters = {"num_samples": num_samples, "mapping": mapping, "lamda": lamda}
    for file1 in os.listdir("models/checkpoints/SMALLAE/MNIST/"):
        if file1.endswith(".pth") and "_0.01_128_20_1" in file1:
            model1.load_state_dict(torch.load("models/checkpoints/SMALLAE/MNIST/" + file1))

            # Sample images
            images_sampled_equally, labels_sampled_equally, indices_sampled_equally = sample_equally_per_class_images(num_samples, images, labels)
            print(f"Sampled {num_samples} images per class")
        
            for file2 in os.listdir("models/checkpoints/SMALLAE/MNIST/"):
                if file1 != file2 and file2.endswith(".pth") and "_0.01_128_20_2" in file2:
                    model2.load_state_dict(torch.load("models/checkpoints/SMALLAE/MNIST/" + file2))

                    # Get latents
                    latent_left = model1.get_latent_space(images).detach().cpu().numpy()
                    latent_right = model2.get_latent_space(images).detach().cpu().numpy()
                    latent_left_sampled_equally = torch.tensor(model1.get_latent_space(images_sampled_equally).detach().cpu().numpy())
                    latent_right_sampled_equally = torch.tensor(model2.get_latent_space(images_sampled_equally).detach().cpu().numpy())
                    print(f"Loaded latents for {file1} and {file2}")

                    # Create mapping and visualize
                    cfg = Config(**parameters)
                    mapping = create_mapping(cfg, latent_left_sampled_equally, latent_right_sampled_equally)
                    mapping.fit()
                    storage_path = f'results/transformations/'
                    Path(storage_path).mkdir(parents=True, exist_ok=True)
                    mapping.save_results(storage_path + "mapping.pth")

                    transformed_latent_space = mapping.transform(latent_left)
                    _, latents1_2d = visualize_latent_space_pca(latents=latent_left, labels=labels, fig_path= storage_path + "latent_left_sampled_equally.png", alpha=0.7, show_fig=True)
                    pca, latents2_2d = visualize_latent_space_pca(latents=latent_right, labels=labels, fig_path=storage_path + "latent_right_sampled_equally.png",  alpha=0.7, show_fig=True)
                    _, latents1_trafo_2d = visualize_latent_space_pca(transformed_latent_space, labels, storage_path + "latents1_transformed_sampled_equally.png",  pca=pca, alpha=0.7, show_fig=True)

                    # Visualize mapping errors
                    errors = np.linalg.norm(transformed_latent_space - latent_right, axis=1)
                    visualize_mapping_error(latents1_2d, errors, storage_path, show_fig=True)


# %%



