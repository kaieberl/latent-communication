import sys

import numpy as np
import pandas as pd
import torch
from matplotlib import pyplot as plt
from sklearn.decomposition import PCA

sys.path.append("..")
from vit.train_vit import MNISTDataModule, MNISTClassifier


def highlight_cluster(ax, df, target, alpha, cmap, norm, size=5):
    cluster_df = df[df["target"] == target]
    ax.scatter(cluster_df.x, cluster_df.y, c=cmap(norm(cluster_df["target"])), alpha=alpha, s=size)


def plot_latent_space(ax, df, targets, size, cmap, norm, bg_alpha=1, alpha=1):
    ax.scatter(df.x, df.y, c=cmap(norm(df["target"])), alpha=bg_alpha, s=size)
    for target in targets:
        highlight_cluster(ax, df, target, alpha=alpha, size=size, cmap=cmap, norm=norm)
    return ax


def visualize_latent_space_pca(latents, labels, fig_path=None, anchors=None, pca=None, size=10, bg_alpha=1, alpha=1):
    """
    Visualizes the 2D latent space obtained from PCA.

    Args:
        latents: A tensor of shape (N, C, H, W) representing the latent space.
        labels: A tensor of shape (N,) representing the labels for each latent point.
        fig_path: Optional; Path to save the figure.
        anchors: Optional; A tensor of shape (M, C, H, W) representing anchor points in the latent space.
        pca: Optional; A PCA object to use for transforming the latent space.
        size: Optional; Size of the points in the plot.
        bg_alpha: Optional; Alpha value for the background points.
        alpha: Optional; Alpha value for the highlighted points.
    """
    if pca is None:
        pca = PCA(n_components=2)
        latents_2d = pca.fit_transform(latents.view(latents.size(0), -1).cpu().detach().numpy())
    else:
        latents_2d = pca.transform(latents.view(latents.size(0), -1).cpu().detach().numpy())

    # Normalize latents
    latents_2d -= latents_2d.min(axis=0)
    latents_2d /= latents_2d.max(axis=0)

    # Create a DataFrame for easy plotting
    latent_df = pd.DataFrame(latents_2d, columns=['x', 'y'])
    latent_df['target'] = labels.numpy()

    # Plot the 2D latent space
    fig, ax = plt.subplots(figsize=(6, 6))
    cmap = plt.get_cmap('tab10')
    norm = plt.Normalize(latent_df['target'].min(), latent_df['target'].max())

    ax = plot_latent_space(ax, latent_df, targets=labels.unique(), size=size, cmap=cmap, norm=norm, bg_alpha=bg_alpha, alpha=alpha)

    if anchors is not None:
        # plot anchors with star marker
        anchors_2d = pca.transform(anchors.view(anchors.size(0), -1).cpu().detach().numpy())
        ax.scatter(anchors_2d[:, 0], anchors_2d[:, 1], marker='*', s=100, c='black')

    plt.title('2D PCA of Latent Space')
    if fig_path is not None:
        plt.savefig(fig_path)
    plt.show()

    return pca, latents_2d


def visualize_mapping_error(latent1, errors, fig_path=None):
    """
    Visualizes the mapping error between two sets of latent variables.

    Args:
        latent1: A tensor or numpy array of shape (N, 2), representing the first set of 2D latent points.
        errors: A tensor or numpy array of shape (N,), representing the mapping error between the two sets of latent points.
        fig_path: Optional; Path to save the figure.
    """
    if isinstance(latent1, torch.Tensor):
        latent1 = latent1.cpu().detach().numpy()

    assert latent1.shape[0] == errors.shape[0], "Number of latent points and errors must match."

    # Create a DataFrame for plotting
    latent_df = pd.DataFrame(latent1, columns=['x', 'y'])
    latent_df['error'] = errors

    # Plot the positions of latent1 with colors encoding the error
    plt.figure(figsize=(8, 6))
    scatter = plt.scatter(latent_df['x'], latent_df['y'], c=latent_df['error'], cmap='YlOrRd', s=10, vmin=4, vmax=16)
    plt.colorbar(scatter, label='Mapping Error (Euclidean Distance)')
    plt.title('Latent Space with Mapping Error')
    plt.xlabel('Latent Dimension 1')
    plt.ylabel('Latent Dimension 2')

    if fig_path is not None:
        plt.savefig(fig_path)
    plt.show()


def setup_model(seed, device):
    model = MNISTClassifier().to(device)
    model.load_state_dict(torch.load(f"models/vit_mnist_seed{seed}_new.pth"))
    model.eval()
    return model


def save_latent_space(model, dataloader, prefix, seed):
    latents, labels = model.get_latent_space_from_dataloader(dataloader)
    torch.save(labels, f"models/labels_{prefix}.pt")
    torch.save(latents, f"models/latent_space_vit_seed{seed}_{prefix}.pt")


def load_and_visualize_latent_space():
    labels = torch.load(f"models/labels_test.pt", map_location='cpu')
    latents1 = torch.load(f"models/latent_space_vit_seed1_test.pt", map_location='cpu')
    latents2 = torch.load(f"models/latent_space_vit_seed0_test_translated.pt", map_location='cpu')
    print(f"Mean error for linear mapping: {np.mean(np.linalg.norm(latents1.detach().numpy() - latents2.detach().numpy(), axis=1)):.4f}")
    pca, latents1_2d = visualize_latent_space_pca(latents1, labels, "figures/latent_space_pca_vit_seed1_test.png")
    errors = np.linalg.norm(latents1.detach().numpy() - latents2.detach().numpy(), axis=1)
    _, latents2 = visualize_latent_space_pca(latents2, labels, "figures/latent_space_pca_vit_seed0_test_linear.png", pca=pca)
    visualize_mapping_error(latents1_2d, errors, "figures/mapping_error_vit_seed0_seed1_test_linear.png")

    latents2 = torch.load(f"models/latent_space_vit_seed0_test_affine.pt", map_location='cpu')
    print(f"Mean error for affine mapping: {np.mean(np.linalg.norm(latents1.detach().numpy() - latents2.detach().numpy(), axis=1)):.4f}")
    errors = np.linalg.norm(latents1.detach().numpy() - latents2.detach().numpy(), axis=1)
    _, latents2 = visualize_latent_space_pca(latents2, labels, "figures/latent_space_pca_vit_seed0_test_affine.png", pca=pca)
    visualize_mapping_error(latents1_2d, errors, "figures/mapping_error_vit_seed0_seed1_test_affine.png")


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data_module = MNISTDataModule(data_dir=".", batch_size=128)

    target_model = setup_model(1, device)

    # For validation set
    save_latent_space(target_model, data_module.val_dataloader(), "test", seed=1)
    load_and_visualize_latent_space()

    # For training set
    save_latent_space(target_model, data_module.train_dataloader(), "train", seed=1)


if __name__ == "__main__":
    main()
