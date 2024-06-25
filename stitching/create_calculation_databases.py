"""Invoke with:
python stitching/create_calculation_databases.py --config-name config_calc -m directory_to_explore=results/transformations/mapping_files/PCKTAE filters=FMNIST.convex hydra.output_subdir=null output_name=PCKTAE

This script creates the calculation databases for the transformations. It takes the mapping files and the results of the models and calculates four different dataframes to then use for various visualizations. The script is called from the command line and takes the following arguments:
    - directory_to_explore: The directory where the mapping files are stored.
    - filters: The filters to apply to the files in the directory.
    - output_name: The name of the output files.
"""

import os
from pathlib import Path

import hydra
import hydra.core.global_hydra
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from omegaconf import DictConfig
from tqdm import tqdm

from utils.dataloaders.get_dataloaders import define_dataloader
from utils.get_mapping import load_mapping
from utils.model import load_model

DEVICE = (
    torch.device("cuda")
    if torch.cuda.is_available()
    else torch.device("mps") if torch.backends.mps.is_available() else "cpu"
)
hydra.core.global_hydra.GlobalHydra.instance().clear()


def find_latentcommunication_dir():
    """
    Finds the 'latent-communication' directory in the directory tree.

    Returns:
        str: The absolute path of the 'latent-communication' directory.

    Raises:
        FileNotFoundError: If the 'latent-communication' folder was not found in the directory tree.
    """
    script_path = os.path.abspath(__file__)
    current_dir = os.path.dirname(script_path)
    while current_dir and os.path.basename(current_dir) != "latent-communication":
        current_dir = os.path.dirname(current_dir)
    if os.path.basename(current_dir) == "latent-communication":
        return current_dir
    else:
        raise FileNotFoundError(
            "The 'latent-communication' folder was not found in the directory tree."
        )


try:
    latentcommunication_dir = find_latentcommunication_dir()
    os.chdir(latentcommunication_dir)
    print(f"Changed working directory to: {latentcommunication_dir}")
except FileNotFoundError as e:
    print(e)


def create_datasets(filters, directory_to_explore):
    """
    Create datasets based on the given filters and directory to explore.

    Args:
        filters (str): A string of filters separated by '.'.
        directory_to_explore (str): The directory path to explore.

    Returns:
        None
    """
    filters = filters.split(".")
    results_list_explore = sorted(os.listdir(directory_to_explore))
    results_list_classes = []
    results_top = []
    error_distribution = []

    # Initialize old data information to avoid repeated loading
    data_info_1_old, data_info_2_old, name_dataset1_old = None, None, None

    # Loss criterion
    criterion = nn.MSELoss()
    list_va = [file for file in results_list_explore if all(x in file for x in filters)]
    list_va = sorted(list_va)
    # Loopcount
    iteration = tqdm(list_va, desc="Processing files", position=0, leave=True)

    top_indices_model1, top_indices_model2, low_indices_model2, low_indices_model1 = (
        [],
        [],
        [],
        [],
    )
    # Loop through files and process
    for file in iteration:
        file = file[:-4]
        iteration.set_description(f"Processing {file}")
        data_info_1, data_info_2, trans_info = file.split(">")

        # Load dataset and model 1 if needed
        if name_dataset1_old != data_info_1.split("_")[0]:
            name_dataset1, name_model1, size_of_the_latent1, seed1 = data_info_1.split(
                "_"
            )
            images, labels, n_classes = define_dataloader(
                name_dataset1, name_model1, seed=seed1, use_test_set=True
            )
            images, labels = images.to(DEVICE).float(), labels.to(DEVICE)
            class_indices = {
                i: np.where(labels.cpu().numpy() == i)[0] for i in range(n_classes)
            }
            indices_class = {i: labels.cpu().numpy()[i] for i in range(len(labels))}
            images_np = images.detach().cpu().numpy()

        if data_info_1_old != data_info_1:
            name_dataset1, name_model1, size_of_the_latent1, seed1 = data_info_1.split(
                "_"
            )
            file1 = f"models/checkpoints/{name_model1}/{name_dataset1}/{name_dataset1}_{name_model1}_{size_of_the_latent1}_{seed1}.pth"
            model1 = load_model(
                model_name=name_model1,
                name_dataset=name_dataset1,
                latent_size=size_of_the_latent1,
                seed=seed1,
                model_path=file1,
            ).to(DEVICE)
            latent_left = model1.get_latent_space(images).detach().to(DEVICE).float()
            latent_left_np = latent_left.detach().cpu().numpy()
            decoded_left = model1.decode(latent_left).to(DEVICE).float()
            decoded_left_np = decoded_left.detach().cpu().numpy()
            best_images_model1 = np.mean(
                np.abs(decoded_left_np - images_np), axis=tuple(range(1, images.ndim))
            )
            sorted_indices_model1 = np.argsort(best_images_model1)
            num_top_indices = int(np.ceil(best_images_model1.size * 0.05))
            top_indices_model1 = sorted_indices_model1[-num_top_indices:]
            low_indices_model1 = sorted_indices_model1[:num_top_indices]
            mse_loss_model1_single = criterion(decoded_left, images).item()
            mse_loss_model1 = []
            mse_loss_per_image = np.mean(
                (decoded_left_np - images_np) ** 2, axis=(1, 2, 3)
            )
            for i in range(n_classes):
                indices = class_indices[i]
                mse_loss_model1.append(
                    criterion(decoded_left[indices], images[indices]).item()
                )
                class_curr = str(i)
                mean = np.mean(mse_loss_per_image[indices])
                variance = np.var(mse_loss_per_image[indices])

                error_distribution.append(
                    {
                        "model": data_info_1,
                        "parent_left": None,
                        "parent_right": None,
                        "class": class_curr,
                        "mean": mean,
                        "variance": variance,
                    }
                )

        if data_info_2_old != data_info_2:
            name_dataset2, name_model2, size_of_the_latent2, seed2 = data_info_2.split(
                "_"
            )
            file2 = f"models/checkpoints/{name_model2}/{name_dataset2}/{name_dataset2}_{name_model2}_{size_of_the_latent2}_{seed2}.pth"
            model2 = load_model(
                model_name=name_model2,
                name_dataset=name_dataset2,
                latent_size=size_of_the_latent2,
                seed=seed2,
                model_path=file2,
            ).to(DEVICE)
            latent_right = model2.get_latent_space(images).to(DEVICE).float()
            latent_right_np = latent_right.detach().cpu().numpy()
            decoded_right = model2.decode(latent_right).to(DEVICE).float()
            decoded_right_np = decoded_right.detach().cpu().numpy()
            best_images_model2 = np.mean(
                np.abs(decoded_right_np - images_np), axis=tuple(range(1, images.ndim))
            )
            sorted_indices_model2 = np.argsort(best_images_model2)
            top_indices_model2 = sorted_indices_model2[-num_top_indices:]
            low_indices_model2 = sorted_indices_model2[:num_top_indices]
            latent_diff_original = np.abs(
                np.sum(latent_right.cpu().detach().numpy(), axis=1)
                - np.sum(latent_left.cpu().detach().numpy(), axis=1)
            )
            mse_loss_model2_single = criterion(decoded_right, images).item()
            mse_loss_model2 = []
            mse_loss_per_image = np.mean(
                (decoded_right_np - images_np) ** 2, axis=(1, 2, 3)
            )
            for i in range(n_classes):
                indices = class_indices[i]
                mse_loss_model2.append(
                    criterion(decoded_right[indices], images[indices]).item()
                )
                class_curr = str(i)
                mean = np.mean(mse_loss_per_image[indices])
                variance = np.var(mse_loss_per_image[indices])

                error_distribution.append(
                    {
                        "model": data_info_2,
                        "parent_left": None,
                        "parent_right": None,
                        "class": class_curr,
                        "mean": mean,
                        "variance": variance,
                    }
                )
        Flag_transfom_latent = 0
        if (data_info_1_old != data_info_1) or (data_info_2_old != data_info_2):
            if latent_right_np.shape[1] < latent_left_np.shape[1]:
                # Add zeros to the latent_right
                size = latent_left_np.shape[1] - latent_right_np.shape[1]
                Flag_transfom_latent = size
                latent_right_enlarged = np.pad(
                    latent_right_np,
                    ((0, 0), (0, size)),
                    mode="constant",
                    constant_values=0,
                )
            else:
                latent_right_enlarged = latent_right_np
            if latent_right_np.shape[1] > latent_left_np.shape[1]:
                # Add zeros to the latent_left
                size = latent_right_np.shape[1] - latent_left_np.shape[1]
                latent_left_enlarged = np.pad(
                    latent_left_np,
                    ((0, 0), (0, size)),
                    mode="constant",
                    constant_values=0,
                )
            else:
                latent_left_enlarged = latent_left_np

        # Process transformations
        list_info_trans = trans_info.split("_")
        mapping_name, num_samples, lamda_t = (
            list_info_trans.pop(0),
            list_info_trans.pop(0),
            list_info_trans.pop(0),
        )
        sampling_strategy = "_".join(list_info_trans)
        mapping = load_mapping(directory_to_explore + "/" + file, mapping_name)
        transformed_latent_space = torch.tensor(
            mapping.transform(latent_left), dtype=torch.float32
        ).to(DEVICE)

        # Normalize the latent space
        latent_right_normalized = (
            latent_right_enlarged
            / np.linalg.norm(latent_right_enlarged, axis=1)[:, np.newaxis]
        )
        latent_left_normalized = (
            latent_left_enlarged
            / np.linalg.norm(latent_left_enlarged, axis=1)[:, np.newaxis]
        )

        latent_transformed_normalized = (
            transformed_latent_space.detach().cpu().numpy()
            / np.linalg.norm(transformed_latent_space.detach().cpu().numpy(), axis=1)[
                :, np.newaxis
            ]
        )
        if Flag_transfom_latent != 0:
            latent_transformed_normalized_enlarged = np.pad(
                latent_transformed_normalized,
                ((0, 0), (0, Flag_transfom_latent)),
                mode="constant",
                constant_values=0,
            )
            Flag_transfom_latent
        else:
            latent_transformed_normalized_enlarged = latent_transformed_normalized

        # Calculate the cosine similarity
        cosine_similarity_original = np.sum(
            latent_right_normalized * latent_left_normalized, axis=1
        )
        cosine_similarity_stitched_mod1 = np.sum(
            latent_left_normalized * latent_transformed_normalized_enlarged, axis=1
        )
        cosine_similarity_stitched_mod2 = np.sum(
            latent_right_normalized * latent_transformed_normalized_enlarged, axis=1
        )
        # Decode latents
        decoded_transformed = model2.decode(transformed_latent_space).to(DEVICE).float()
        # Calculate reconstruction errors
        decoded_transformed_np = decoded_transformed.detach().cpu().numpy()
        best_images_stitched = np.mean(
            np.abs(decoded_transformed_np - images_np),
            axis=tuple(range(1, images.ndim)),
        )
        mse_loss_per_image = np.mean(
            (decoded_transformed_np - images_np) ** 2, axis=(1, 2, 3)
        )

        # Get indices of top and bottom 5% Images
        sorted_indices_stitched = np.argsort(best_images_stitched)
        top_indices_stitched = sorted_indices_stitched[-num_top_indices:]
        low_indices_stitched = sorted_indices_stitched[:num_top_indices]

        # Latent differences (eh, this makes sense only if the latent sizes are the same)
        latent_diff_stitched_mod1 = np.abs(
            np.sum(latent_right.cpu().detach().numpy(), axis=1)
            - np.sum(transformed_latent_space.cpu().detach().numpy(), axis=1)
        )
        latent_diff_stitched_mod2 = np.abs(
            np.sum(latent_left.cpu().detach().numpy(), axis=1)
            - np.sum(transformed_latent_space.cpu().detach().numpy(), axis=1)
        )

        # Record top and bottom indices information
        for i in range(len(labels)):
            (
                model1_top,
                model2_top,
                stitched_top,
                model1_low,
                model2_low,
                stitched_low,
            ) = (False, False, False, False, False, False)
            if i in top_indices_model1:
                model1_top = True
            if i in top_indices_model2:
                model2_top = True
            if i in top_indices_stitched:
                stitched_top = True
            if i in low_indices_model1:
                model1_low = True
            if i in low_indices_model2:
                model2_low = True
            if i in low_indices_stitched:
                stitched_low = True

            results_top.append(
                {
                    "dataset": name_dataset1,
                    "model1": file1,
                    "model2": file2,
                    "sampling_strategy": sampling_strategy,
                    "mapping": mapping_name,
                    "lambda": lamda_t,
                    "num_samples": num_samples,
                    "reconstruction_error_model1": best_images_model1[i],
                    "reconstruction_error_model2": best_images_model2[i],
                    "reconstruction_error_stitched": best_images_stitched[i],
                    "latent_diff_original": np.linalg.norm(latent_diff_original[i]),
                    "latent_diff_mod1": np.linalg.norm(latent_diff_stitched_mod1[i]),
                    "latent_diff_mod2": np.linalg.norm(latent_diff_stitched_mod2[i]),
                    "class": indices_class[i],
                    "model1_top": model1_top,
                    "model2_top": model2_top,
                    "stitched_top": stitched_top,
                    "model1_low": model1_low,
                    "model2_low": model2_low,
                    "stitched_low": stitched_low,
                    "cosine_similarity_original": cosine_similarity_original[i],
                    "cosine_similarity_stitched_mod1": cosine_similarity_stitched_mod1[
                        i
                    ],
                    "cosine_similarity_stitched_mod2": cosine_similarity_stitched_mod2[
                        i
                    ],
                }
            )

        # Calculate class-wise MSE losses
        for i in range(n_classes):
            indices = class_indices[i]
            mse_loss_class = criterion(
                decoded_transformed[indices], images[indices]
            ).item()

            results_list_classes.append(
                {
                    "dataset": name_dataset1,
                    "model1": file1,
                    "model2": file2,
                    "sampling_strategy": sampling_strategy,
                    "mapping": mapping_name,
                    "lambda": lamda_t,
                    "latent_dim": size_of_the_latent2,
                    "num_samples": num_samples,
                    "MSE_loss": mse_loss_class,
                    "MSE_loss_model1": mse_loss_model1[i],
                    "MSE_loss_model2": mse_loss_model2[i],
                    "class": i,
                }
            )

    results_top_df = pd.DataFrame(results_top)
    results_class_df = pd.DataFrame(results_list_classes)
    error_distribution_df = pd.DataFrame(error_distribution)
    return results_top_df, results_class_df, error_distribution_df


@hydra.main(version_base=None, config_path="../config", config_name="config")
def main(cfg: DictConfig) -> None:
    cfg.base_dir = Path(hydra.utils.get_original_cwd()).parent
    results_top_df, results_class_df, error_distribution_df = (
        create_datasets(cfg.filters, cfg.directory_to_explore)
    )
    results_top_df.to_csv(
        "results/transformations/calculations_databases/" + cfg.output_name + "_top.csv"
    )
    results_class_df.to_csv(
        "results/transformations/calculations_databases/"
        + cfg.output_name
        + "_class.csv"
    )
    error_distribution_df.to_csv(
        "results/transformations/calculations_databases/"
        + cfg.output_name
        + "_error_distribution.csv"
    )


if __name__ == "__main__":
    main()
