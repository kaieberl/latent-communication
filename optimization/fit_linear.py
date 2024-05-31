"""Load two pre-trained models and perform linear and affine fitting between their latent spaces, or between two sets
of latent vectors."""

import os

import torch

from stitching.stitching import get_transformations, load_model
from optimizer import AffineFitting, LinearFitting
from utils.sampler import simple_sampler

# Set device
device = torch.device('cuda') if torch.cuda.is_available() else 'cpu'

# Configuration
base_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
seed1 = 0
seed2 = 1

config = {
    'model1': {
        # 'path': os.path.join(base_path, f"vit/models/vit_mnist_seed{seed1}.pth"),
        'label_path': os.path.join(base_path, "vit/models/labels_train.pth"),
        'latents_path': os.path.join(base_path, f"vit/models/latent_space_vit_seed{seed1}_train.pth"),
        'name': 'vit',
        'seed': seed1
    },
    'model2': {
        # 'path': os.path.join(base_path, f"vit/models/vit_mnist_seed{seed2}.pth"),
        'latents_path': os.path.join(base_path, f"vit/models/latent_space_vit_seed{seed2}_train.pth"),
        'name': 'vit',
        'seed': seed2
    }
}


def get_latents(config):
    """For both models, load the latent vectors if latent_path is provided, else load the models and sample the
    latent vectors."""

    torch.manual_seed(0)
    indices = torch.randperm(60000)[:1000]
    if 'latents_path' in config:
        labels = torch.load(config['label_path'], map_location=device)
        latents1 = torch.load(config['latents_path'], map_location=device)
        latents2 = torch.load(config['latents_path'], map_location=device)
        latents1 = latents1[indices]
        latents2 = latents2[indices]
    else:
        model1 = load_model(config['modelname1'], config['path1'])
        model2 = load_model(config['modelname2'], config['path2'])

        transforms1 = get_transformations(config['modelname1'])
        transforms2 = get_transformations(config['modelname2'])

        latents1 = simple_sampler(indices, model1, transforms1, device, seed=seed1)
        latents2 = simple_sampler(indices, model2, transforms2, device, seed=seed2)

    return latents1, latents2


def main():
    latents1, latents2 = get_latents(config)

    # Linear transform
    linear_fitting = LinearFitting(latents1, latents2, lamda=0.01)
    linear_fitting.solve_problem()
    linear_fitting.print_results()
    linear_fitting.save_results("results/linear_fitting.npz")

    # Affine transform
    affine_fitting = AffineFitting(latents1, latents2, lamda=0.01)
    affine_fitting.solve_problem()
    affine_fitting.print_results()
    affine_fitting.save_results("results/affine_fitting.npz")


if __name__ == '__main__':
    main()
