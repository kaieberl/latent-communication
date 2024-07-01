"""loads PCKTAE with different latent sizes and plots the reconstruction loss"""

import os

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

from tqdm import tqdm

from utils.dataloaders.full_dataloaders import DataLoaderMNIST
from utils.model import load_model, get_transformations, get_reconstruction_error


def get_errors(latent_sizes, test_loader, folder, epochs=None):
    errors = []
    folder = Path(folder)
    for d in tqdm(latent_sizes):
        for seed in [1, 2, 3]:
            if epochs:
                file = f"FMNIST_PCKTAE_{d}_{seed}_{epochs}.pth"
            else:
                file = f"FMNIST_PCKTAE_{d}_{seed}.pth"
            model = load_model(model_name="pcktae", name_dataset="fmnist", latent_size=d,
                                model_path=folder / file)
            errors.append({
                "latent_size": d,
                "seed": seed,
                "MSE": get_reconstruction_error(model, test_loader)
            })

    return pd.DataFrame(errors)


if __name__ == '__main__':
    base_dir = Path(os.getcwd()).parent.parent
    data_loader_model = DataLoaderMNIST(128, get_transformations("pcktae"), seed=0, base_path=base_dir)
    test_loader = data_loader_model.get_test_loader()

    folder = base_dir / "models/checkpoints/PCKTAE/FMNIST"
    latent_sizes = [2, 4, 6, 8, 10, 30, 50]
    print("Calculating errors for 15 epochs")
    errors = get_errors(latent_sizes, test_loader, folder)
    print("Calculating errors for 50 epochs")
    errors_50 = get_errors(latent_sizes, test_loader, folder, epochs=50)

    # plot reconstruction errors against latent size
    sns.lineplot(errors, x='latent_size', y='MSE', markers='o')
    sns.lineplot(errors_50, x='latent_size', y='MSE', markers='o', color='red')
    plt.show()