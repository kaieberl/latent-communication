import numpy as np
import torch
from utils.dataloaders.dataloader_mnist_single import DataLoaderMNIST
import math
import torch.nn.functional as F


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

def collect_dataset(data_loader, batch_size, transformations, seed):
    dataset = []
    labels = []
    data_loader = data_loader(batch_size, transformations, seed=seed)
    train_loader = data_loader.get_train_loader()
    for _, images, labels_batch in train_loader:
        dataset.extend(images)
        labels.extend(labels_batch)
    return torch.stack(dataset), torch.tensor(labels)

def sample_equally_per_class(n_samples, data_loader, batch_size=128, transformations=[], seed=0):
    dataset, labels = collect_dataset(data_loader, batch_size, transformations, seed)
    label_to_images = {label.item(): [] for label in torch.unique(labels)}
    
    for image, label in zip(dataset, labels):
        label_to_images[label.item()].append(image)

    n_per_class = n_samples // len(label_to_images)
    images_sampled = []
    labels_sampled = []
    
    for label, images in label_to_images.items():
        indices = np.random.choice(len(images), n_per_class, replace=False)
        images_sampled.extend([images[i] for i in indices])
        labels_sampled.extend([label] * n_per_class)

    return torch.stack(images_sampled), torch.tensor(labels_sampled)


def sample_with_half_best_classes(n_samples, model, data_loader, batch_size=128, transformations=[], seed=0, loss_fun=F.mse_loss, device='cpu'):
    dataset, labels = collect_dataset(data_loader, batch_size, transformations, seed)
    label_to_images = {label.item(): [] for label in torch.unique(labels)}
    losses_by_class = {label.item(): [] for label in torch.unique(labels)}
    
    for image, label in zip(dataset, labels):
        label = label.item()
        image = image.to(device)
        x = model(image.unsqueeze(0))
        loss = loss_fun(x, image.unsqueeze(0)).item()
        label_to_images[label].append(image)
        losses_by_class[label].append(loss)
    
    best_classes = sorted(losses_by_class, key=lambda x: np.mean(losses_by_class[x]))[len(losses_by_class)//2:]
    
    label_to_images = {k: v for k, v in label_to_images.items() if k in best_classes}
    n_per_class = n_samples // len(label_to_images)
    images_sampled = []
    labels_sampled = []
    
    for label, images in label_to_images.items():
        indices = np.random.choice(len(images), n_per_class, replace=False)
        images_sampled.extend([images[i] for i in indices])
        labels_sampled.extend([label] * n_per_class)

    return torch.stack(images_sampled), torch.tensor(labels_sampled)


def sample_with_half_worst_classes(n_samples, model, data_loader, batch_size=128, transformations=[], seed=0, loss_fun=F.mse_loss, device='cpu'):
    dataset, labels = collect_dataset(data_loader, batch_size, transformations, seed)
    label_to_images = {label.item(): [] for label in torch.unique(labels)}
    losses_by_class = {label.item(): [] for label in torch.unique(labels)}
    
    for image, label in zip(dataset, labels):
        label = label.item()
        image = image.to(device)
        x = model(image.unsqueeze(0))
        loss = loss_fun(x, image.unsqueeze(0)).item()
        label_to_images[label].append(image)
        losses_by_class[label].append(loss)
    
    worst_classes = sorted(losses_by_class, key=lambda x: np.mean(losses_by_class[x]))[:len(losses_by_class)//2]
    
    label_to_images = {k: v for k, v in label_to_images.items() if k in worst_classes}
    n_per_class = n_samples // len(label_to_images)
    images_sampled = []
    labels_sampled = []
    
    for label, images in label_to_images.items():
        indices = np.random.choice(len(images), n_per_class, replace=False)
        images_sampled.extend([images[i] for i in indices])
        labels_sampled.extend([label] * n_per_class)

    return torch.stack(images_sampled), torch.tensor(labels_sampled)


###
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