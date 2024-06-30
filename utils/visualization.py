from pathlib import Path

import numpy as np
import pandas as pd
import torch
from matplotlib import pyplot as plt
from sklearn.decomposition import PCA
import seaborn as sns
from sklearn.manifold import TSNE


def highlight_cluster(ax, df, target, alpha, cmap, norm, size=5):
    cluster_df = df[df["target"] == target]
    ax.scatter(cluster_df.x, cluster_df.y, c=cmap(norm(cluster_df["target"])), alpha=alpha, s=size)


def plot_latent_space(ax, df, targets, size, cmap, norm, bg_alpha=1, alpha=1):
    ax.scatter(df.x, df.y, c=cmap(norm(df["target"])), alpha=bg_alpha, s=size)
    for target in targets:
        highlight_cluster(ax, df, target, alpha=alpha, size=size, cmap=cmap, norm=norm)
    return ax


def visualize_latent_space(latents, labels, fig_path=None, anchors=None, trafo=None, size=10, bg_alpha=1, alpha=1,
                           title=None, show_fig=True, mode='pca'):
    """
    Visualizes the 2D latent space obtained from PCA.

    Args:
        latents: A tensor of shape (N, dim) representing the latent points.
        labels: A tensor of shape (N,) representing the labels for each latent point.
        fig_path: Optional; Path to save the figure.
        anchors: Optional; A tensor of shape (M, dim) representing anchor points in the latent space.
        trafo: Optional; A PCA object to use for transforming the latent space.
        size: Optional; Size of the points in the plot.
        bg_alpha: Optional; Alpha value for the background points.
        alpha: Optional; Alpha value for the highlighted points.
    """
    if isinstance(latents, torch.Tensor):
        latents = latents.detach().cpu().numpy()
    if isinstance(labels, torch.Tensor):
        labels = labels.detach().cpu().numpy()
    if trafo is None:
        if mode == 'pca':
            trafo = PCA(n_components=2)
            latents_2d = trafo.fit_transform(latents)
        elif mode == 'tsne':
            trafo = TSNE(n_components=2)
            latents_2d = trafo.fit_transform(latents)
    else:
        latents_2d = trafo.transform(latents)

    # Normalize latents
    minimum = latents_2d.min(axis=0)
    maximum = latents_2d.max(axis=0)

    latents_2d = (latents_2d - minimum) / (maximum - minimum)

    # Create a DataFrame for easy plotting
    latent_df = pd.DataFrame(latents_2d, columns=['x', 'y'])
    latent_df['target'] = labels

    # Plot the 2D latent space
    fig, ax = plt.subplots(figsize=(6, 6))
    cmap = plt.get_cmap('tab10')
    norm = plt.Normalize(latent_df['target'].min(), latent_df['target'].max())

    ax = plot_latent_space(ax, latent_df, targets=np.unique(labels), size=size, cmap=cmap, norm=norm, bg_alpha=bg_alpha,
                           alpha=alpha)

    if anchors is not None:
        # plot anchors with star marker
        anchors_2d = trafo.transform(anchors.view(anchors.size(0), -1).cpu().detach().numpy())
        anchors_2d = (anchors_2d - minimum) / (maximum - minimum)
        ax.scatter(anchors_2d[:, 0], anchors_2d[:, 1], marker='*', s=10, c='black')

    if title:
        ax.set_title(title)

    if fig_path:
        plt.savefig(fig_path)
    if show_fig:
        plt.show()

    return trafo, latents_2d


def visualize_mapping_error(latent1, errors, fig_path=None, show_fig=True):
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
    if show_fig:
        plt.show()


def plot_error_dist(errors, labels, fig_path=None):
    """
    Plots the distribution of mapping errors.

    Args:
        errors: A tensor or numpy array of shape (N,), representing the mapping errors.
        labels: A tensor or numpy array of shape (N,), representing the labels for each latent point.
        mapping: A string representing the type of mapping used.
        fig_path: Optional; Path to save the figure.
    """
    if isinstance(errors, torch.Tensor):
        errors = errors.numpy()
    for label in np.unique(labels):
        sns.kdeplot(errors[labels == label], label=f'Class {label}', fill=True)
    plt.title('Mapping Error Distribution')
    plt.xlabel('Mapping Error')
    plt.ylabel('Density')
    plt.legend()
    if fig_path is not None:
        plt.savefig(fig_path)
    plt.show()


def visualize_results(cfg, labels, latents1, latents2):
    """
    Creates two scatter plots of the input latent spaces and a scatter plot of the error between the two latent spaces, as well as a histogram of the error distribution by class.

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

    Path(cfg.storage_path).mkdir(parents=True, exist_ok=True)
    pca, latents1_2d = visualize_latent_space(latents1, labels,
                                                  f"{cfg.storage_path}/latent_space_pca_{cfg.model2.name}_{cfg.model2.seed}.png",
                                              show_fig=False)
    visualize_latent_space(latents2, labels,
                               f"{cfg.storage_path}/latent_space_pca_{cfg.model1.name}_{cfg.model1.seed}_transformed.png",
                           trafo=pca, show_fig=False)
    visualize_mapping_error(latents1_2d, errors,
                            f"{cfg.storage_path}/mapping_error_{cfg.model1.name}_{cfg.model1.seed}_{cfg.model2.name}_{cfg.model2.seed}.png",
                            show_fig=False)
    print(f"MSE: {np.mean(errors):.4f}")
    # plot_error_dist(errors, labels, f"{cfg.storage_path}/mapping_error_distribution_{cfg.model1.name}_{cfg.model1.seed}_{cfg.model2.name}_{cfg.model2.seed}.png")


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
    latents_2_2d = latents_2d[len(latents_trans):]

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
