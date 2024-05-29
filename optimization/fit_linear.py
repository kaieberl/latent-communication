import os

import numpy as np
import torch
import torchvision.transforms as transforms

from helper.dataloader_mnist import DataLoaderMNIST
from optimizer import AffineFitting, LinearFitting
from utils.sampler import simple_sampler
from vit.train_vit import MNISTClassifier

# Set device
device = torch.device('cuda') if torch.cuda.is_available() else 'cpu'

# Configuration
base_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
seed1 = 0
seed2 = 1
num_samples = 100
storage_path = 'transforms'

config = {
    'path1': os.path.join(base_path, f"vit/models/vit_mnist_seed{seed1}_new.pth"),
    'modelname1': 'vit',
    'seed1': f'{seed1}',
    'path2': os.path.join(base_path, f"vit/models/vit_mnist_seed{seed2}_new.pth"),
    'modelname2': 'vit',
    'seed2': f'{seed2}',
    'num_samples': num_samples,
    'storage_path': storage_path
}

# Data transformations
transform_pipeline = [
    transforms.ToTensor(),
    transforms.Resize((224, 224)),
    transforms.Lambda(lambda x: x.repeat(3, 1, 1))
]

# DataLoader
data_loader = DataLoaderMNIST(128, transform_pipeline, transform_pipeline)


# Initialize models and load weights
def load_model(path):
    model = MNISTClassifier().to(device)
    model.load_state_dict(torch.load(path, map_location=device))
    return model


model1 = load_model(config['path1'])
model2 = load_model(config['path2'])

# Sampling
z1, z2, images, labels = simple_sampler(num_samples, model1, model2, data_loader, device)


# Linear transformation
def perform_linear_fitting(z1, z2, lamda, config):
    linear_fitting = LinearFitting(z1, z2, lamda)
    linear_fitting.solve_problem()
    name = f"Linear_{config['modelname1']}_{config['seed1']}_{config['modelname2']}_{config['seed2']}_{config['num_samples']}"
    path = os.path.join('optimization', config['storage_path'], name)
    linear_fitting.save_results(path)
    _, A = linear_fitting.get_results()
    np.save(path, A)


perform_linear_fitting(z1, z2, lamda=0.01, config=config)


# Affine transformation
def perform_affine_fitting(z1, z2, lamda, config):
    affine_fitting = AffineFitting(z1, z2, lamda)
    affine_fitting.solve_problem()
    name = f"Affine_{config['modelname1']}_{config['seed1']}_{config['modelname2']}_{config['seed2']}_{config['num_samples']}"
    path = os.path.join('optimization', config['storage_path'], name)
    affine_fitting.save_results(path)
    _, A, b = affine_fitting.get_results()
    np.savez(path, A=A, b=b)


perform_affine_fitting(z1, z2, lamda=0.01, config=config)
