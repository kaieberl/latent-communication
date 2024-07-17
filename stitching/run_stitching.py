"""Script for generating the box and line plots used in the presentation.

Specify the experiment parameters in config/config_plot.yaml, then invoke with:
    python run_stitching.py

The config file contains a `combinations` and `filters` section.
- With the default config file, all mapping parameter combinations for pcktae will be calculated and saved to results.csv, which takes quite long.
This is only done the first time, in successive runs the values will be read from the csv.
If you know in advance that you don't need some mapping combinations, you can exclude them from this section.
- The filters section then specifies granularly which experiments to plot.
"""

import itertools
import logging
import os
from pathlib import Path

import hydra
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from optimization.fit_mapping import get_params_from_model_name, get_mapping_name, get_dataset, get_test_latents
from utils.dataloaders.full_dataloaders import (
    DataLoaderMNIST, DataLoaderFashionMNIST, DataLoaderCIFAR10, DataLoaderCIFAR100
)
from utils.get_mapping import load_mapping
from utils.model import load_model, get_transformations

DEVICE = torch.device("cuda") if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"


def get_dataloader(name, augmentation, batch_size, seed):
    dataloaders = {
        "mnist": DataLoaderMNIST,
        "fmnist": DataLoaderFashionMNIST,
        "cifar10": DataLoaderCIFAR10,
        "cifar100": DataLoaderCIFAR100
    }
    return dataloaders[name.lower()](transformation=augmentation, batch_size=batch_size, seed=seed)


def load_full_dataset(dataloader, test):
    if test:
        return dataloader.get_full_test_dataset()
    else:
        return dataloader.get_full_train_dataset()


def define_dataloader(file, file2, test=False):
    if file.strip("_")[0] != file2.strip("_")[0]:
        logging.error("The datasets are different")
    dataset_name, model_name, latent_size, seed = file.strip(".pth").split("_")
    augmentation = get_transformations(model_name)
    dataloader = get_dataloader(dataset_name, augmentation, 64, int(seed))
    return load_full_dataset(dataloader, test) + (len(np.unique(dataloader.get_full_train_dataset()[1].numpy())),)


def plot_loss(filters, results, file_path=None):
    """For each element in filters_dict, create a box plot and line plot with the loss.

    Args:
        filters (dict): dictionary with filters.
        results (list): list of dictionaries with results.
        file_path (str): path to save the plots, e.g. path/to/FMNIST_VAE. Then _mse_comparison.png and _mse_overview.png will be appended.
    """
    filter_results = []
    for filter_dict in filters:
        name = filter_dict['name']
        filter_dict = {key: value for key, value in filter_dict.items() if key != 'name'}
        for result in results:
            if all(result[key] == value for key, value in filter_dict.items()):
                result['name'] = name
                filter_results.append(result)

    means = {}
    for name in [filter['name'] for filter in filters]:
        means[name] = np.mean([result['MSE_loss'] for result in filter_results if result['name'] == name])
    print(means)

    filter_results = pd.DataFrame(filter_results)

    # Plot MSE_loss vs. mapping (as a categorical variable)
    plt.figure(figsize=(8, 5))
    sns.boxplot(data=filter_results, x='name', y='MSE_loss')
    # plt.xticks(rotation=45)
    plt.xlabel('Mapping')
    plt.ylabel('Reconstruction Error')
    plt.ylim(bottom=0)
    plt.tight_layout()
    if file_path:
        plt.savefig(file_path + '_mse_comparison.png', dpi=300)
    plt.show()

    # Plot mean MSE_loss vs. latent dimension, make them appear successively
    reference = means.pop('Linear')
    for i in range(len(means) + 1):
        plt.plot(list(means.keys())[:i], list(means.values())[:i], 'o-')
        # disable x ticks
        plt.xticks([])
        plt.xlabel('Mapping')
        plt.ylabel('Reconstruction Error')
        plt.axhline(y=reference, color='tab:orange', linestyle='--')
        plt.xlim([-0.25, len(means) - 0.75])
        plt.ylim([0, max([*means.values(), reference]) * 1.1])
        plt.tight_layout()
        if file_path:
            plt.savefig(file_path + f'_mse_overview_{i}.png', dpi=300)
        plt.show()


def get_model_files(folder):
    return [f for f in os.listdir(folder) if f.endswith(".pth")]


def compute_losses(cfg, model1, model2, mapping, criterion=None):
    if criterion is None:
        criterion = nn.MSELoss()
    latent_left, latent_right, _ = get_test_latents(cfg)
    latent_left = latent_left.to(DEVICE)
    latent_right = latent_right.to(DEVICE)
    transformed_latent_space = mapping.transform(latent_left)

    decoded_left = model1.decode(latent_left).detach()
    decoded_right = model2.decode(latent_right).detach()
    decoded_transformed = model2.decode(transformed_latent_space.to(torch.float32).to(DEVICE)).detach()

    dataset = get_dataset(cfg, cfg.model1.name, train=False)

    dataloader = DataLoader(dataset, batch_size=len(dataset), shuffle=False)
    for data in dataloader:
        images, _ = data
        images = images.to(DEVICE)
        mse_loss = criterion(decoded_transformed, images).item()
        mse_loss_model1 = criterion(decoded_left, images).item()
        mse_loss_model2 = criterion(decoded_right, images).item()

    return mse_loss, mse_loss_model1, mse_loss_model2


def process_combinations(cfg, folder1, folder2):
    if (Path(cfg.mapping_dir) / cfg.model1.name.upper() / "results.csv").exists():
        results = pd.read_csv(Path(cfg.mapping_dir) / cfg.model1.name.upper() / "results.csv")
        results_list = results.to_dict(orient="records")
        # remove columns with Unnamed
        results_list = [{key: value for key, value in result.items() if "Unnamed: 0" not in key} for result in results_list]
        # only keep results where mapping != Adaptive
        # results_list = [result for result in results_list if not result["mapping"] == "Adaptive"]
    else:
        results_list = []

    model_files1 = get_model_files(folder1)
    model_files2 = get_model_files(folder2)

    combinations = []
    for combination_dict in cfg.combinations:
        mapping_param_list = itertools.product(*combination_dict.values())
        combinations.extend(list(itertools.product(model_files1, model_files2, mapping_param_list)))
    # check if files are different and have the same latent dimension
    combinations = [c for c in combinations if c[0] != c[1] and c[0].split("_")[2] == c[1].split("_")[2]]

    for file1, file2, (num_samples, mapping_name, lamda, dropout, noisy, hidden_size, sampling, latent_dim) in tqdm(combinations):
        # if instance is already in results_list, skip
        fields = {
            "model1": file1,
            "model2": file2,
            "mapping": mapping_name,
            "lambda": lamda,
            "dropout": dropout,
            "noisy": noisy,
            "hidden_size": hidden_size,
            "sampling": sampling,
            "num_samples": num_samples,
            "latent_dim": latent_dim
        }
        if any(all(result[field] == fields[field] for field in fields) for result in results_list):
            continue

        dataset1, model_name1, latent_size1, seed1 = get_params_from_model_name(file1)
        dataset2, model_name2, latent_size2, seed2 = get_params_from_model_name(file2)

        if latent_dim != latent_size1 or dataset1.lower() != cfg.dataset:
            continue

        model1 = load_model(model_name=model_name1, name_dataset=dataset1, latent_size=int(latent_size1), seed=int(seed1),
                            model_path=folder1 / file1)
        model2 = load_model(model_name=model_name2, name_dataset=dataset2, latent_size=int(latent_size2), seed=int(seed2),
                            model_path=folder2 / file2)

        mapping_path = get_mapping_name(cfg.dataset, model_name1, latent_size1, seed1, model_name2, latent_size2, seed2, mapping_name, num_samples, lamda, sampling, dropout, noisy, hidden_size)
        try:
            mapping = load_mapping(Path(cfg.mapping_dir) / model_name2 / mapping_path, mapping_name)
        except FileNotFoundError:
            continue

        cfg.model1.path = folder1 / file1
        cfg.model1.latent_size = latent_size1
        cfg.model2.path = folder2 / file2
        cfg.model2.latent_size = latent_size2
        mse_loss, mse_loss_model1, mse_loss_model2 = compute_losses(cfg, model1, model2, mapping)

        # check for duplicates
        new_result = {
            "dataset": cfg.dataset,
            "model1": file1,
            "model2": file2,
            "mapping": mapping_name,
            "lambda": lamda,
            "dropout": dropout,
            "noisy": noisy,
            "hidden_size": hidden_size,
            "sampling": sampling,
            "num_samples": num_samples,
            "MSE_loss": mse_loss,
            "latent_dim": latent_size1,
            "MSE_loss_model1": mse_loss_model1,
            "MSE_loss_model2": mse_loss_model2
        }
        fields = new_result.keys()
        if not any(all(result[field] == new_result[field] for field in fields) for result in results_list):
            results_list.append(new_result)

    return results_list


@hydra.main(version_base="1.1", config_path="../config", config_name="config_plot")
def main(cfg):
    cfg.base_dir = Path(hydra.utils.get_original_cwd()).parent
    folder1 = cfg.base_dir / "models/checkpoints" / cfg.model1.name.upper() / cfg.dataset.upper()
    folder2 = cfg.base_dir / "models/checkpoints" / cfg.model2.name.upper() / cfg.dataset.upper()

    results_list = process_combinations(cfg, folder1, folder2)
    results = pd.DataFrame(results_list)
    results.to_csv(Path(cfg.mapping_dir) / cfg.model1.name.upper() / "results.csv", index=False)
    plot_loss(cfg.filters, results_list, str(Path(cfg.mapping_dir) / f'mse_comparison'))


if __name__ == "__main__":
    main()
