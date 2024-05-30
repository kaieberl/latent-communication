# Import relevant libraries
import numpy as np
import matplotlib.pyplot as plt
import torch
import torchvision.transforms as transforms
import os
import sys
sys.path.append('../')
from utils.dataloaders.DataLoaderMNIST_single import DataLoader_MNIST

# Configuration
seed1 = 3
seed2 = 4

# 12, 13, 14, 23, 24 ,34
config = {
    'path1': os.path.dirname(os.getcwd())+f"/models/checkpoints/ResNet/MNIST/model_seed{seed1}.pth",
    'modelname1': 'resnet',
    'seed1': f'{seed1}',
    'path2': os.path.dirname(os.getcwd())+f"/models/checkpoints/ResNet/MNIST/model_seed{seed2}.pth",
    'modelname2': 'resnet',
    'seed2': f'{seed2}',
    'num_samples': '100',
    'storage_path': 'ResNet-LinearTransform'

}



def load_model(model_name, model_path):
    DEVICE = torch.device('cpu')
    
    if model_name == 'VAE':
        from models.definitions.vae import VAE
        model = VAE(in_dim=784, dims=[256, 128, 64, 32], distribution_dim=16).to(DEVICE)
    elif model_name == 'resnet':
        from models.definitions.resnet import ResNet
        model = ResNet().to(DEVICE)
    else:
        raise ValueError(f"Unknown model name: {model_name}")
    
    model.load_state_dict(torch.load(model_path, map_location=DEVICE))
    return model

def load_Models():
    model1 = load_model(config['modelname1'], config['path1'])
    model2 = load_model(config['modelname2'], config['path2'])
    return model1, model2

def get_transformations(model_name):
    if model_name == 'VAE':
        return [
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,)),
            transforms.Lambda(lambda x: x.view(-1))
        ]
    elif model_name == 'resnet':
        return [
            transforms.ToTensor(),
            transforms.Lambda(lambda x: x.repeat(3, 1, 1))
        ]
    else:
        raise ValueError(f"Unknown model name: {model_name}")

def transformations():
    transformations1 = get_transformations(config['modelname1'])
    transformations2 = get_transformations(config['modelname2'])
    return transformations1, transformations2

def get_accuracy(model,test_loader):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for data in test_loader:
            images, labels = data
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    # Calculate accuracy
    accuracy = 100 * correct / total
    return accuracy

# Load models
model1, model2 = load_Models()

# Get transformations
transformations1, transformations2 = transformations()

# Initialize data loader
data_loader_model = DataLoader_MNIST(128, get_transformations(config['modelname1']), seed=10)
test_loader = data_loader_model.get_test_loader()
# Print metrics for model 1 on test set with torch.accuracy
accuracy1 = get_accuracy(model1, test_loader)
accuracy2 = get_accuracy(model2, test_loader)
print('Accuracy of model 1 on the test images: %.2f %%' % accuracy1)
print('Accuracy of model 2 on the test images: %.2f %%' % accuracy2)

#Print metrics for model 2

#TODO stitching and evaluating metrics