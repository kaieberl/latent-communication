# Workspace Overview

This workspace is structured to support a Python-based machine learning or deep learning project, leveraging PyTorch and PyTorch Lightning for model development and training. It includes configurations for various neural network architectures and setups, data handling, and optimization processes.

## Structure

The workspace is organized into several directories and files, each serving a specific purpose in the project lifecycle:

- `.dvc/`, `.dvcignore`: Directories and files related to Data Version Control (DVC), used for managing and versioning datasets and machine learning models.
- `.gitignore`: Specifies intentionally untracked files to ignore.
- `config/`: Houses YAML configuration files for different models and experiments.
- `data/`: Intended for storing datasets used in the project.
- `models/`: Contains Python scripts for model definitions and training routines.
- `optimization/`, `results/`: Directories for storing optimization processes and final results respectively. Here the mappings Linear, Affine, Decouple, MLP, MLP + Linear 
can be calculated. The mappings are specified in optimization/optimizer.py. They can be calculated in fit_mapping.py with hydra or for running multiple configuarations (different lamda, number of samples, combination between models) optimization/optimization_run_all.ipynb can be used. The mappings are stored in the results/mapping_files/{name_model3} with name convention '{name_filenamemodel1}>{name_filenamemodel2}>{mapping}_{num_samples}_{lamda}_{sampling_stragey}' e.g. 'FMNIST_PCKTAEClASS01234_10_1>FMNIST_PCKTAEClASS01234_10_2>Affine_100_0.001_equally'.
- `setup.py`: A setup script for installing the project as a package.
- `stitching/': In this script the stitching can be performed with the mapping previously calculated in optimization. In stitching/stitching_run_all.ipynb the stitching for all models specified in one folder is performed. The performance is measured in a csv file. Then you can plot dependent on the number of samples, lambda, mapping type and also analyse the behavior for different classes. The file stitching/create_calculation_databases.ipynb takes mapping files and
the results of the models and calculates four different dataframes to then use for various visualizations.
-  `utils/`: Include utility scripts and modules for data processing, model loading, sampling strategys and other helper functions.

## Key Components

- **Model Definitions and Training**: The `models/` directory contains scripts like `models/definitions/vit.py` and `models/train.py` (models/training_models) for defining and training neural network models.
- **Configuration Management**: Various YAML files in the `config/` directory allow for flexible configuration of models and experiments.
- **Mapping Creation**: The `optimization/optimization_run_all.ipynb` creates all the mappings (for now, linear, affine and NN) between the models saved in one folder.
- **Data Processing**: The script `stitching/create_calculation_databases.py` handles data processing and preparation for visualization, creating three databases from which one can get different information regarding the performance of the different mapping under different settings.
- **Visualization**: The notebook `stitching/visualize_perfomances.ipynb` takes as input these databases, and generates various plots for future visualization


## Installation

To set up this project, ensure you have Python and PyTorch installed, then follow these steps:

1. Clone the repository to your local machine.
2. Navigate to the project directory and install Python dependencies with `pip install -r requirements.txt`.
3. Set up DVC for data and model versioning as needed.

## Usage

Usage instructions vary depending on the specific task or experiment. Generally, you would:

1. Configure your experiment using the YAML files in the `config/` directory.
2. Run model training or data processing scripts, adjusting parameters as necessary.
3. Analyze results in the `results/` directory or through generated visualizations.


## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for more details.
