import torch 
import numpy as np


def simple_sampler(m,model1,model2, data_loader, DEVICE):
    """
    Input: 
    - m: Samples 
    - model1: Model 1
    - model2: Model 2
    Output:
    - z1: Latent vectors of the model 1
    - z2: Latent vectors of the model 2

    This function samples the latent space of the model and returns the latent vectors of the model 1 and model2
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
    
     # Sample indices from the train set
    indices = np.random.choice(all_images.shape[0], m, replace=False)

    all_images_sample = all_images[indices]
    all_labels_sample = all_labels[indices]
    

    z1 = model1.get_latent_space(all_images_sample.to(DEVICE))
    z2 = model2.get_latent_space(all_images_sample.to(DEVICE))

    # Detach from GPU
    z1 = z1.detach().cpu().numpy()
    z2 = z2.detach().cpu().numpy()  

    return z1, z2, all_images_sample, all_labels_sample


def class_sampler(m, model1, model2, data_loader, DEVICE):
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
    z1 = model1.get_latent_space(all_images_sample.to(DEVICE))
    z2 = model2.get_latent_space(all_images_sample.to(DEVICE))

    # Detach from GPU
    z1 = z1.detach().cpu().numpy()
    z2 = z2.detach().cpu().numpy()

    return z1, z2, all_images_sample, all_labels_sample