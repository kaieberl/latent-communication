import numpy as np
import pandas as pd
import torch
from matplotlib import pyplot as plt
from sklearn.decomposition import PCA


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
        latents: A tensor of shape (N, dim) representing the latent points.
        labels: A tensor of shape (N,) representing the labels for each latent point.
        fig_path: Optional; Path to save the figure.
        anchors: Optional; A tensor of shape (M, dim) representing anchor points in the latent space.
        pca: Optional; A PCA object to use for transforming the latent space.
        size: Optional; Size of the points in the plot.
        bg_alpha: Optional; Alpha value for the background points.
        alpha: Optional; Alpha value for the highlighted points.
    """
    if isinstance(latents, torch.Tensor):
        latents = latents.detach().cpu().numpy()
    if isinstance(labels, torch.Tensor):
        labels = labels.detach().cpu().numpy()

    if pca is None:
        pca = PCA(n_components=2)
        latents_2d = pca.fit_transform(latents)
    else:
        latents_2d = pca.transform(latents)

    # Normalize latents
    latents_2d -= latents_2d.min(axis=0)
    latents_2d /= latents_2d.max(axis=0)

    # Create a DataFrame for easy plotting
    latent_df = pd.DataFrame(latents_2d, columns=['x', 'y'])
    latent_df['target'] = labels

    # Plot the 2D latent space
    fig, ax = plt.subplots(figsize=(6, 6))
    cmap = plt.get_cmap('tab10')
    norm = plt.Normalize(latent_df['target'].min(), latent_df['target'].max())

    ax = plot_latent_space(ax, latent_df, targets=np.unique(labels), size=size, cmap=cmap, norm=norm, bg_alpha=bg_alpha, alpha=alpha)

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
        latent1 = latent1.detach().numpy()
    if isinstance(errors, torch.Tensor):
        errors = errors.numpy()

    assert latent1.shape[0] == errors.shape[0], "Number of latent points and errors must match."

    # Create a DataFrame for plotting
    latent_df = pd.DataFrame(latent1, columns=['x', 'y'])
    latent_df['error'] = errors

    # Plot the positions of latent1 with colors encoding the error
    plt.figure(figsize=(8, 6))
    scatter = plt.scatter(latent_df['x'], latent_df['y'], c=latent_df['error'], cmap='YlOrRd', s=10)
    plt.colorbar(scatter, label='Mapping Error (Euclidean Distance)')
    plt.title('Latent Space with Mapping Error')
    plt.xlabel('Latent Dimension 1')
    plt.ylabel('Latent Dimension 2')

    if fig_path is not None:
        plt.savefig(fig_path)
    plt.show()


def visualize_results(cfg, labels, latents1, latents2):
    """
    Creates two scatter plots of the input latent spaces and a scatter plot of the error between the two latent spaces.

    Args:
        cfg (DictConfig): Configuration dictionary
        labels (torch.Tensor): Labels
        latents1 (torch.Tensor): Target latent vectors
        latents2 (torch.Tensor): Transformed source latent vectors

    Returns:
        None
    """
    if isinstance(latents1, torch.Tensor):
        latents1 = latents1.detach().cpu().numpy()
    if isinstance(latents2, torch.Tensor):
        latents2 = latents2.detach().cpu().numpy()
    errors = np.linalg.norm(latents1 - latents2, axis=1)

    pca, latents1_2d = visualize_latent_space_pca(latents1, labels,
                                                  f"{cfg.storage_path}/latent_space_pca_{cfg.model2.name}_{cfg.model2.seed}.png")
    visualize_latent_space_pca(latents2, labels,
                               f"{cfg.storage_path}/latent_space_pca_{cfg.model1.name}_{cfg.model1.seed}_transformed.png",
                               pca=pca)
    visualize_mapping_error(latents1_2d, errors,
                            f"{cfg.storage_path}/mapping_error_{cfg.model1.name}_{cfg.model1.seed}_{cfg.model2.name}_{cfg.model2.seed}.png")
    print(f"MSE: {np.mean(errors):.4f}")
