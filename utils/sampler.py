import numpy as np
import torch
from utils.dataloaders.dataloader_mnist_single import DataLoaderMNIST


def simple_sampler(indices, model, transformations, device, seed=10):
    """
    Input:
    - model: Model 
    - indices: Indices of the dataset
    - transformations: Transformations to be applied to the images
    Output:
    - z: Latent vectors of the model
    - labels: Labels of the dataset

    This function samples the latent space of the model and returns the latent vectors
    """
    data_loader = DataLoaderMNIST(128, transformations, seed=seed, indices=indices)
    train_loader = data_loader.get_train_loader()

    #get all images from train_loader and convert them to latent space
    all_images = []
    for images, _ in train_loader:
        images = images.to(device)
        latent_space = model.get_latent_space(images)
        all_images.append(latent_space)

    z = torch.cat(all_images, dim=0)
    z = z.detach().cpu().numpy()
    return z, data_loader.train_loader.dataset.dataset.targets[indices]

import numpy as np
import torch

def simple_sampler_v1(samples, model, transformations, device, seed=10):
    """
    Input:
    - model: Model 
    - samples: Number of samples
    - transformations: Transformations to be applied to the images
    Output:
    - z: Latent vectors of the model
    - labels: Labels of the dataset

    This function samples the latent space of the model and returns the latent vectors
    """
    data_loader = DataLoaderMNIST(128, transformations, seed=seed)
    train_loader = data_loader.get_train_loader()

    # Get all images from train_loader and convert them to latent space
    all_images = []
    all_labels = []
    for images, labels in train_loader:
        images = images.to(device)
        latent_space = model.get_latent_space(images)
        latent_space = latent_space.detach().cpu()
        all_images.append(latent_space)
        all_labels.extend(labels.cpu().numpy())
    
    all_images = torch.cat(all_images, dim=0)
    all_labels = np.array(all_labels)

    # Sample the latent space
    indices = np.random.choice(len(all_images), samples, replace=False)
    z = all_images[indices]
    labels = all_labels[indices]

    return z.numpy(), labels

# Assuming DataLoaderMNIST and model are defined elsewhere in your code


def exluding_sampler_v1(m, labels, model, data_loader, device):
    """
    Input:
    - m: Samples
    - labels: List of labels to exclude
    - model: Model
    Output:
    - z: Latent vectors of the model

    This function samples the latent space of the model and returns the latent vectors
    We exclude the labels from the list
    """
    images, _ = next(iter(data_loader.train_loader))
    all_images = []
    all_labels = []
    for images, labels in data_loader.train_loader:
        all_images.append(images)
        all_labels.append(labels)
    # Concatenate all the batches to form a single tensor for images and labels
    all_images = torch.cat(all_images, dim=0)
    all_labels = torch.cat(all_labels, dim=0)

    # Exclude labels
    indices = []
    for label in labels:
        indices_label = np.where(all_labels != label)[0]
        indices.extend(indices_label)
    
    all_images_sample = all_images[indices]
    all_labels_sample = all_labels[indices]
    # Get latent space 
    z = model.get_latent_space(all_images_sample.to(device))

    # Detach from GPU
    z = z.detach().cpu().numpy()

    return z, all_labels_sample


def class_sampler(m, model1, model2, data_loader, device):
    """
    Input:
    - m: Samples
    - model1: Model 1
    - model2: Model 2
    Output:
    - z1: Latent vectors of the model 1
    - z2: Latent vectors of the model 2

    This function samples the latent space of the model and returns the latent vectors of the model 1 and model2
    We sample m//len(labels) samples from each class
    """
    images, _ = next(iter(data_loader.train_loader))
    all_images = []
    all_labels = []
    for images, labels in data_loader.train_loader:
        all_images.append(images)
        all_labels.append(labels)
    # Concatenate all the batches to form a single tensor for images and labels
    all_images = torch.cat(all_images, dim=0)
    all_labels = torch.cat(all_labels, dim=0)

    # Distinct labels
    labels = torch.unique(all_labels)
    # Sample size per label
    m_per_label = m // len(labels)
    # Sample from each label
    indices = []
    for label in labels:
        indices_label = np.where(all_labels == label)[0]
        indices_label = np.random.choice(indices_label, m_per_label, replace=False)
        indices.extend(indices_label)
    
    all_images_sample = all_images[indices]
    all_labels_sample = all_labels[indices]
    # Get latent space 
    z1 = model1.get_latent_space(all_images_sample.to(device))
    z2 = model2.get_latent_space(all_images_sample.to(device))

    # Detach from GPU
    z1 = z1.detach().cpu().numpy()
    z2 = z2.detach().cpu().numpy()

    return z1, z2