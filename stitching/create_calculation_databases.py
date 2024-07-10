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
    
    
def save_dataframes(results_list_classes, output_name):
    """
    Save the dataframes to the results folder.

    Args:
        results_top_df (pd.DataFrame): The dataframe containing the top indices information.
        results_class_df (pd.DataFrame): The dataframe containing the class-wise MSE losses.
        error_distribution_df (pd.DataFrame): The dataframe containing the error distribution information.
        output_name (str): The name of the output files.

    Returns:
        None
    """
    results_class_df = pd.DataFrame(results_list_classes)
    #check if the file already exists
    if os.path.exists("results/transformations/calculations_databases/" + output_name + "_top.csv"):

        results_class_df.to_csv(
            "results/transformations/calculations_databases/" + output_name + "_class.csv",
            mode="a",
            header=False, sep="#"
        )
    
    else:

        results_class_df.to_csv(
            "results/transformations/calculations_databases/" + output_name + "_class.csv", sep="#"
        )

    return []

def create_old_datasets(directory_to_explore):
    '''
    Create the old datasets based on the given directory to explore.
    '''
    dataframe = pd.read_csv(directory_to_explore)
    #drop all columns that are not names
    dataframe = dataframe[['model1', 'model2', 'sampling_strategy', 'mapping', 'num_samples', 'lambda']]
    #dataframe join the columns toghtether in a single column of strings
    dataframe['combined'] = dataframe[['sampling_strategy', 'mapping', 'num_samples', 'lambda']].astype(str).agg('_'.join, axis=1) 
    dataframe = dataframe[['model1', 'model2','combined']].astype(str).agg('>'.join, axis=1)
    return dataframe
    
    
def criterion(prediction, images):
    with torch.no_grad():
        errors = nn.MSELoss(reduction='none')(prediction, images)  # Assign loss to 'errors'
        errors = torch.mean(errors, dim=(1, 2, 3))  # Average across channels and spatial dimensions
        return errors


def create_datasets(filters, directory_to_explore, current_dir, output_name):
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

#    old_list = create_old_datasets(current_dir)

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
    

    
    count = 0
    # Loop through files and process
    for file in iteration:
        file = file[:-4]
        torch.no_grad()
        iteration.set_description(f"Processing {file}")
        data_info_1, data_info_2, trans_info = file.split(">")
        mean_1, variance_1, mean_2, variance_2 = [], [], [], []
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
            latent_left_np = latent_left.detach().cpu().numpy()
            decoded_left = model1.decode(latent_left).to(DEVICE).float()
            decoded_left_np = decoded_left.detach().cpu().numpy()
            #calculate all the errors
            errors_by_image_model_1 = criterion(decoded_left, images).detach().cpu().numpy()
            #couple each error with its index
            sorted_indices_model1 = np.argsort(errors_by_image_model_1)
            
            num_top_indices = int(np.ceil(errors_by_image_model_1.size * 0.01))
            top_indices_model1, low_indices_model1 =  [], []
            for i in range(n_classes):
                indices = class_indices[i]
                filtered_indices = [idx for idx in sorted_indices_model1 if idx in indices]
                top_indices_model1.append([idx for idx in filtered_indices[-num_top_indices:]])
                low_indices_model1.append([idx for idx in filtered_indices[:num_top_indices]])
                mean_1.append(np.mean(errors_by_image_model_1[indices]))
                variance_1.append(np.var(errors_by_image_model_1[indices]))

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
            latent_right_np = latent_right.detach().cpu().numpy()
            decoded_right = model2.decode(latent_right).to(DEVICE).float()
            decoded_right_np = decoded_right.detach().cpu().numpy()
            errors_by_image_model_2 = criterion(decoded_right, images).detach().cpu().numpy()
            #couple each error with its index
            sorted_indices_model2 = np.argsort(errors_by_image_model_2)
            
            num_top_indices = int(np.ceil(errors_by_image_model_2.size * 0.03))
            top_indices_model2, low_indices_model2 =  [], []
            for i in range(n_classes):
                indices = class_indices[i]
                filtered_indices = [idx for idx in sorted_indices_model2 if idx in indices]
                top_indices_model2.append([idx for idx in filtered_indices[-num_top_indices:]])
                low_indices_model2.append([idx for idx in filtered_indices[:num_top_indices]])
                mean_2.append(np.mean(errors_by_image_model_2[indices]))
                variance_2.append(np.var(errors_by_image_model_2[indices]))

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


        # Decode latents
        decoded_transformed = model2.decode(transformed_latent_space).to(DEVICE).float()
        # Calculate reconstruction errors
        decoded_transformed_np = decoded_transformed.detach().cpu().numpy()
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
        if count%50 == 0:
            results_list_classes = save_dataframes(
                results_list_classes, output_name=output_name
            )
        count += 1
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
        create_datasets(cfg.filters, cfg.directory_to_explore, cfg.path_to_test_file, cfg.output_name)
    )
    
    results_class_df = pd.DataFrame(results_class_df)

    results_class_df.to_csv(
        "results/transformations/calculations_databases/"
        + cfg.output_name
        + "_class.csv", mode="a", header=False, sep="#"
    )


if __name__ == "__main__":
    main()
