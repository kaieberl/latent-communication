"""Invoke with:
python stitching/create_calculation_databases.py --config-name config_calc -m directory_to_explore=results/transformations/mapping_files/PCKTAE filters=FMNIST.convex hydra.output_subdir=null output_name=PCKTAE

This script creates the calculation databases for the transformations. It takes the mapping files and
the results of the models and calculates four different dataframes to then use for various visualizations. 
The script is called from the command line and takes the following arguments:
    - directory_to_explore: The directory where the mapping files are stored.
    - filters: The filters to apply to the files in the directory. They are separated by '.'.
    - output_name: The name of the output files. BE SURE TO CHANGE IT FOR EACH RUN TO AVOID OVERWRITING FILES.
"""
import os
import sys
from pathlib import Path
import hydra
import hydra.core.global_hydra
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from omegaconf import DictConfig
from tqdm import tqdm

# Define the project root directory name
PROJECT_ROOT_DIR = "latent-communication"

# Get the absolute path of the current script
current_dir = os.path.abspath(os.path.dirname(__file__))

# Find the project root by walking up the directory tree
while current_dir:
    if os.path.basename(current_dir) == PROJECT_ROOT_DIR:
        break  # Found the project root!
    current_dir = os.path.dirname(current_dir)
else:
    raise FileNotFoundError(f"Project root '{PROJECT_ROOT_DIR}' not found in the directory tree.")

# Add the project root and any necessary subdirectories to sys.path
sys.path.insert(0, current_dir) 
sys.path.insert(0, os.path.join(current_dir, "utils"))  # Add the utils directory if needed


from utils.dataloaders.get_dataloaders import define_dataloader
from utils.get_mapping import load_mapping
from utils.model import load_model

DEVICE = (
    torch.device("cuda")
    if torch.cuda.is_available() else
    'cpu' if torch.backends.mps.is_available()
    else "cpu"
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
    
    
def criterion(prediction, images):
    with torch.no_grad():
        errors = nn.MSELoss(reduction='none')(prediction, images)  # Assign loss to 'errors'
        if images.size()[-1] == 28:
            errors = torch.mean(errors, dim=(1, 2, 3)) 
        if images.size()[-1] == 784:
            errors = torch.mean(errors, dim=1)
        return errors


def create_datasets(filters, directory_to_explore, output_name):
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

    # Initialize old data information to avoid repeated loading
    data_info_1_old, data_info_2_old, name_dataset1_old = None, None, None

    # Old_list = create_old_datasets(current_dir)

    list_va = [file for file in results_list_explore if all(x in file for x in filters)]
    list_va = sorted(list_va)
    # Loopcount
    iteration = tqdm(list_va, desc="Processing files", position=0, leave=True)

    # Loop through files and process
    for file in iteration:
        file = file[:-4]
        torch.no_grad()
        iteration.set_description(f"Processing {file}")
        data_info_1, data_info_2, trans_info = file.split(">")
        # This is actually just to check if the Ddataset has changed (which is unlikely)
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
        # This checks if the first model has to be changes, and will recaluclate the latent space
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
            ).to(DEVICE).eval()
            latent_left = model1.get_latent_space(images).float()
            decoded_left = model1.decode(latent_left).to(DEVICE).float()
            errors_by_image_model_1 = criterion(decoded_left, images).detach().cpu().numpy()

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
            ).to(DEVICE).eval()
            latent_right = model2.get_latent_space(images).to(DEVICE).float()
            decoded_right = model2.decode(latent_right).to(DEVICE).float()
            errors_by_image_model_2 = criterion(decoded_right, images).detach().cpu().numpy()
            
        # Process transformations
        list_info_trans = trans_info.split("_")
        mapping_name, num_samples, lamda_t = (
            list_info_trans.pop(0),
            list_info_trans.pop(0),
            list_info_trans.pop(0),
        )
        sampling_strategy = "_".join(list_info_trans)
        mapping = load_mapping(directory_to_explore + "/" + file, mapping_name)
        transformed_latent_space = mapping.transform(latent_left).clone().detach().to(DEVICE).to(torch.float32)


        # Decode latents
        decoded_transformed = model2.decode(transformed_latent_space).to(DEVICE).float()
        # Calculate reconstruction errors
        errors_by_image_stiched = criterion(decoded_transformed, images).detach().cpu().numpy()
        
        # Record top and bottom indices information
        for i in range(n_classes):
            indices = class_indices[i]

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
                    "MSE_loss": np.mean(errors_by_image_stiched[indices]),
                    "MSE_loss_model1": np.mean(errors_by_image_model_1[indices]),
                    "MSE_loss_model2": np.mean(errors_by_image_model_2[indices]),
                    "MSE_variance": np.var(errors_by_image_stiched[indices]),
                    "MSE_variance_model1": np.var(errors_by_image_model_1[indices]),
                    "MSE_variance_model2": np.var(errors_by_image_model_2[indices]),
                    "class": i,
                }
            )

    return results_list_classes


@hydra.main(version_base=None, config_path="../config", config_name="config")
def main(cfg: DictConfig) -> None:
    """
    Main function for creating calculation databases.

    Args:
        cfg (DictConfig): Configuration dictionary.

    Returns:
        None
    """
    cfg.base_dir = Path(hydra.utils.get_original_cwd()).parent
    results_class_df = (
        create_datasets(cfg.filters, cfg.directory_to_explore, cfg.output_name)
    )
    
    results_class_df = pd.DataFrame(results_class_df)

    results_class_df.to_csv(
        "results/transformations/calculations_databases/"
        + cfg.output_name
        + "_class.csv", header=False, sep="#"
    )


if __name__ == "__main__":
    main()
