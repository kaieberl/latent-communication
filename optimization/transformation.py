from pathlib import Path
import torch
from omegaconf import DictConfig
import hydra
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST
from torchvision.transforms import transforms

from utils.model import load_model, get_transformations
from utils.sampler import *

def fit_transformation(sampler, lamda, model1, model2, mapping):
    ''' Fits the transformation using given sampler
    Args:
        sampler: Sampler function 
        lamda: Regularization parameter
        model1: Model 1
        model2: Model 2
        mapping: 'Linear', 'Affine'
    '''
    images, labels = sampler

    z1 = model1.get_latent_space(images)
    z2 = model2.get_latent_space(images)

    if(mapping == 'Linear'):
        from optimization.optimizer import LinearFitting
        mapping = LinearFitting(z1, z2, lamda)
    elif(mapping == 'Affine'):
        from optimization.optimizer import AffineFitting
        mapping = AffineFitting(z1, z2, lamda)
    
    return mapping
