import numpy as np
import torch

from vit.train_vit import MNISTClassifier


def simple_sampler(m, model1, model2, data_loader, device):
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
    data_loader1, data_loader2 = data_loader.get_train_loader()

    # Sample indices from the train set
    indices = torch.randperm(len(data_loader1.dataset))[:m]

    # Get the corresponding images
    all_images_sample1 = torch.stack([data_loader1.dataset[i][0] for i in indices])
    all_images_sample2 = torch.stack([data_loader2.dataset[i][0] for i in indices])

    z1 = model1.get_latent_space(all_images_sample1.to(device))
    z2 = model2.get_latent_space(all_images_sample2.to(device))

    return z1, z2


def vit_simple_sampler(m, ckpt_path1, ckpt_path2, data_loader, device):
    """
    Input:
    - m: Samples
    - ckpt_path1: Checkpoint path of the model 1
    - ckpt_path2: Checkpoint path of the model 2
    Output:
    - z1: Latent vectors of the model 1
    - z2: Latent vectors of the model 2

    This function samples the latent space of the model and returns the latent vectors of the model 1 and model2
    """
    data_loader1, data_loader2 = data_loader.get_train_loader()

    # Sample indices from the train set
    indices = torch.randperm(len(data_loader1.dataset))[:m]

    # Get the corresponding images
    all_images_sample1 = torch.stack([data_loader1.dataset[i][0] for i in indices])

    z = []

    for ckpt_path in [ckpt_path1, ckpt_path2]:
        model = MNISTClassifier().to(device)
        model.load_state_dict(torch.load(ckpt_path, map_location=device))
        z.append(model.get_latent_space(all_images_sample1.to(device)))
        del model

    return z[0], z[1]


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