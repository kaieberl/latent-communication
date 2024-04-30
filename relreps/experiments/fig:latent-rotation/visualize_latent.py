import pandas as pd
import torch
from matplotlib import pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from torchvision import datasets
from tueplots import figsizes, bundles

CIFAR_ROOT = "../../../0-shot-llm-vision/datasets/cifar-10/validation"


def hightlight_cluster(
    ax,
    df,
    target,
    alpha,
    cmap,
    norm,
    size=0.5,
):
    cluster_df = df[df["target"] == target]
    ax.scatter(cluster_df.x, cluster_df.y, c=cmap(norm(cluster_df["target"])), alpha=alpha, s=size)


def plot_latent_space(ax, df, targets, size, cmap, norm, bg_alpha=0.1, alpha=0.5):
    ax.scatter(df.x, df.y, c=cmap(norm(df["target"])), alpha=bg_alpha, s=size)
    for target in targets:
        hightlight_cluster(ax, df, target, alpha=alpha, size=size, cmap=cmap, norm=norm)
    return ax


def get_latents(model, dataset='cifar10', modality='img', reduction_mode='pca', K=10000):
    """
    Arguments:
        model: name of the model, e.g. vit, convnext, clip, dinov2
        dataset: name of dataset, e.g. cifar10, coco
        modality: img or txt
        reduction_mode: pca or tsne
        K: max number of data points, between 1 and 10000, default: 10000

    Returns:
        df: assembled dataframe with keys x, y, target
        pca: initialized PCA object
    """
    with open(f'../../../0-shot-llm-vision/data/{dataset}_{model}_{modality}.pt', 'rb') as f:
        latents = torch.load(f)
    cap = datasets.CIFAR10(root=CIFAR_ROOT, train=False, download=True)
    targets = []
    for i in range(len(cap)):
        _, target = cap[i]
        targets.append(target)

    if latents.shape[-1] == 2:
        latents2d = latents
    else:
        if reduction_mode == 'pca':
            reduction = PCA(n_components=2)
            reduction.fit(latents)

            latents2d = reduction.transform(latents)
        elif reduction_mode == 'tsne':
            reduction = TSNE(n_components=2, init="pca", learning_rate="auto", random_state=42)
            latents2d = reduction.fit_transform(latents)
        else:
            raise ValueError(f"Unknown reduction mode: {reduction_mode}")

    # normalize latents
    latents2d -= latents2d.min(axis=0)
    latents2d /= latents2d.max(axis=0)

    df = pd.DataFrame(
        {
            "x": latents2d[:K, 0].tolist(),
            "y": latents2d[:K, 1].tolist(),
            "target": targets[:K],
        }
    )
    return df, reduction


if __name__ == '__main__':
    all_latents_df = []
    models = ['vit', 'clip', 'convnext']
    for model_name in models:
        df, _ = get_latents(model_name, reduction_mode='tsne')
        all_latents_df.append(df)

    TO_CONSIDER = range(len(all_latents_df))
    latents_df = [all_latents_df[i] for i in TO_CONSIDER]

    plt.rcParams.update(bundles.icml2022())

    plt.rcParams.update(figsizes.icml2022_full(ncols=len(latents_df), nrows=1, height_to_width_ratio=1.0))
    cmap = plt.cm.get_cmap("Set1", 10)
    norm = plt.Normalize(latents_df[0]["target"].min(), latents_df[0]["target"].max())

    fig, axes = plt.subplots(dpi=150, nrows=1, ncols=len(latents_df), sharey=True, sharex=True, squeeze=True)

    for i, ax in enumerate(axes):
        ax.set_aspect("equal")
        plot_latent_space(
            ax, all_latents_df[i], targets=[0, 2], size=1, cmap=cmap, norm=norm, bg_alpha=0.15
        )

    plt.show()
