import numpy as np
import torch
from utils.dataloaders.full_dataloaders import DataLoaderMNIST
import math
import torch.nn.functional as F
import numpy as np
import torch
import random
import logging

from sklearn.decomposition import PCA
import random
from scipy.spatial.distance import cdist
from scipy.spatial import ConvexHull



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

    # get all images from train_loader and convert them to latent space
    all_images = []
    for images, _ in train_loader:
        images = images.to(device)
        latent_space = model.get_latent_space(images)
        all_images.append(latent_space)

    z = torch.cat(all_images, dim=0)
    z = z.detach().cpu().numpy()
    return z, data_loader.train_loader.dataset.dataset.targets[indices]


def collect_dataset(data_loader, batch_size, transformations, seed):
    dataset = []
    labels = []
    data_loader = data_loader(batch_size, transformations, seed=seed)
    train_loader = data_loader.get_train_loader()
    for images, labels_batch in train_loader:
        dataset.extend(images)
        labels.extend(labels_batch)
    return torch.stack(dataset), torch.tensor(labels)


def get_latents(images, model, device="cpu"):
    model.eval()
    with torch.no_grad():
        latents = []
        for image in images:
            image = image.to(device)
            latent = model(image.unsqueeze(0)).squeeze(0)
            latents.append(latent)
    return torch.stack(latents)


def sample_convex_hulls_images(n_samples, images, labels, model, seed=0, device="cpu", pca_components=3):
    model.eval()

    image_label_dict = {}
    for image, label in zip(images, labels):
        label = label.item()
        if label not in image_label_dict:
            image_label_dict[label] = []
        latent = model.get_latent_space(image.unsqueeze(0).to(device)).detach().cpu().numpy()
        image_label_dict[label].append((image, latent))
    
    n_per_class = math.ceil(n_samples / len(image_label_dict))
    images_sampled = []
    labels_sampled = []
    remaining_samples = 0
    np.random.seed(seed)
    
    for class_name, data in image_label_dict.items():
        images_class, latents_class = zip(*data)
        latents_class = np.array(latents_class).squeeze()  # Squeeze to remove extra dimensions if any
        pca = PCA(n_components=pca_components)
        latents_class_reduced = pca.fit_transform(latents_class)
        
        if len(latents_class_reduced) > 2:  # ConvexHull requires at least 3 points
            convex_hull = ConvexHull(latents_class_reduced)
            vertices = convex_hull.vertices
            
            if len(vertices) >= n_per_class:
                indices = np.random.choice(vertices, n_per_class, replace=False)
                selected_images = [images_class[idx] for idx in indices]
            else:
                logging.info(f"Class {class_name} has less vertices than {n_per_class} samples in the convex hull. Sampling randomly.")
                selected_images = [images_class[idx] for idx in vertices]
                remaining_samples += n_per_class - len(selected_images)
        else:
            logging.info(f"Class {class_name} has less than 3 points, unable to form a convex hull. Sampling randomly.")
            indices = np.random.choice(len(images_class), n_per_class, replace=False)
            selected_images = [images_class[idx] for idx in indices]
            remaining_samples += n_per_class - len(selected_images)
        
        images_sampled.extend(selected_images)
        labels_sampled.extend([class_name] * len(selected_images))
    
    if remaining_samples > 0:
        logging.info(f"Remaining samples: {remaining_samples}")
        remaining_images, remaining_labels = sample_equally_per_class_images(remaining_samples, images, labels, seed)
        images_sampled.extend(remaining_images)
        labels_sampled.extend(remaining_labels)

    return torch.stack(images_sampled), torch.tensor(labels_sampled)

def sample_furthest_away_images(n_samples, images, labels, model, batch_size=128, device='cpu'):

    # Compute latent representations in batches
    latents = []
    for i in range(0, len(images), batch_size):
        batch_images = images[i:i+batch_size].to(device)
        batch_latents = model.get_latent_space(batch_images).detach().cpu().numpy()
        latents.append(batch_latents)
    
    # Concatenate latent representations from batches
    latents = np.concatenate(latents, axis=0)
    
    # Randomly select the first point
    first_point_index = random.randint(0, len(latents) - 1)
    sampled_indices = [first_point_index]
    
    # Iteratively select the next point that maximizes the minimum distance to the already selected points
    for _ in range(1, n_samples):
        remaining_indices = np.setdiff1d(np.arange(len(latents)), sampled_indices)
        max_distances = []
        for i in remaining_indices:
            distances = cdist(latents[i].reshape(1, -1), latents[sampled_indices])
            max_distances.append(np.min(distances))
        next_point_index = remaining_indices[np.argmax(max_distances)]
        sampled_indices.append(next_point_index)
    
    # Return the sampled images, labels, and their indices
    sampled_images = images[sampled_indices]
    sampled_labels = labels[sampled_indices]
    return sampled_images, sampled_labels


def sample_removing_outliers(n_samples, images, labels, model, batch_size=128, device='cpu'):
    model.eval()
    errors = []
    for image in images:
        image = image.to(device)
        x = model(image.unsqueeze(0))
        error = F.mse_loss(x, image.unsqueeze(0)).item()
        errors.append(error)
    mean_error = np.mean(errors)
    variance_error = np.var(errors)
    images = images.detach().cpu().numpy()
    labels = labels.detach().cpu().numpy()
    ## exclude outliers
    filtered_data = [(image, label) for n, (image, label) in enumerate(zip(images, labels)) if errors[n] < variance_error/2 + mean_error]
    filtered_outliers = [image for image, label in filtered_data]
    filtered_outliers_labels = [label for image, label in filtered_data]
    if len(filtered_outliers) < n_samples:
        logging.info(f"Number of images without outliers is less than n_samples.")
        n_samples = len(filtered_outliers)
    filtered_outliers, filtered_outliers_labels =  torch.tensor(filtered_outliers), torch.tensor(filtered_outliers_labels)
    images_sampled, labels_sampled = sample_equally_per_class_images(n_samples, filtered_outliers, filtered_outliers_labels, seed=0)
    return images_sampled, labels_sampled



def sample_equally_per_class_images(n_samples, images, labels, seed=0):
    label_to_images = {label.item(): [] for label in torch.unique(labels)}
    label_to_indices = {label.item(): [] for label in torch.unique(labels)}

    for idx, (image, label) in enumerate(zip(images, labels)):
        label_to_images[label.item()].append(image)
        label_to_indices[label.item()].append(idx)

    n_per_class = math.ceil(n_samples / len(label_to_images))
    images_sampled = []
    labels_sampled = []
    indices_sampled = []
    np.random.seed(seed)
    for label, image_list in label_to_images.items():
        indices = np.random.choice(len(image_list), n_per_class, replace=False)
        images_sampled.extend([image_list[i] for i in indices])
        labels_sampled.extend([label] * n_per_class)
        indices_sampled.extend([label_to_indices[label][i] for i in indices])

    return torch.stack(images_sampled), torch.tensor(labels_sampled)

def sample_with_half_worst_classes_images(n_samples, images, labels, model, loss_fun=F.mse_loss, device="cpu", seed=0):
    label_to_images = {label.item(): [] for label in torch.unique(labels)}
    label_to_indices = {label.item(): [] for label in torch.unique(labels)}
    losses_by_class = {label.item(): [] for label in torch.unique(labels)}
    model.eval()
    for idx, (image, label) in enumerate(zip(images, labels)):
        label = label.item()
        image = image.to(device)
        x = model(image.unsqueeze(0))
        loss = loss_fun(x, image.unsqueeze(0)).item()
        label_to_images[label].append(image)
        label_to_indices[label].append(idx)
        losses_by_class[label].append(loss)

    worst_classes = sorted(losses_by_class, key=lambda x: np.mean(losses_by_class[x]))[
        : len(losses_by_class) // 2
    ]

    label_to_images = {k: v for k, v in label_to_images.items() if k in worst_classes}
    label_to_indices = {k: v for k, v in label_to_indices.items() if k in worst_classes}
    n_per_class = n_samples // len(label_to_images)
    images_sampled = []
    labels_sampled = []
    indices_sampled = []
    np.random.seed(seed)
    for label, image_list in label_to_images.items():
        indices = np.random.choice(len(image_list), n_per_class, replace=False)
        images_sampled.extend([image_list[i] for i in indices])
        labels_sampled.extend([label] * n_per_class)
        indices_sampled.extend([label_to_indices[label][i] for i in indices])

    return torch.stack(images_sampled), torch.tensor(labels_sampled)


def sample_equally_per_class(
    n_samples,
    data_loader,
    batch_size=128,
    transformations=[],
    seed=0,
    using_latents=False,
):
    ## be careful with sampling from two latent spaces

    dataset, labels = collect_dataset(data_loader, batch_size, transformations, seed)
    label_to_images = {label.item(): [] for label in torch.unique(labels)}

    for image, label in zip(dataset, labels):
        label_to_images[label.item()].append(image)

    n_per_class = n_samples // len(label_to_images)
    images_sampled = []
    labels_sampled = []
    np.random.seed(seed)
    for label, images in label_to_images.items():
        indices = np.random.choice(len(images), n_per_class, replace=False)
        images_sampled.extend([images[i] for i in indices])
        labels_sampled.extend([label] * n_per_class)
    print(images_sampled)
    return torch.stack(images_sampled), torch.tensor(labels_sampled)


def sample_with_half_best_classes(
    n_samples,
    model,
    data_loader,
    batch_size=128,
    transformations=[],
    seed=0,
    loss_fun=F.mse_loss,
    device="cpu",
):
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

    best_classes = sorted(losses_by_class, key=lambda x: np.mean(losses_by_class[x]))[
        len(losses_by_class) // 2 :
    ]

    label_to_images = {k: v for k, v in label_to_images.items() if k in best_classes}
    n_per_class = n_samples // len(label_to_images)
    images_sampled = []
    labels_sampled = []
    np.random.seed(seed)
    for label, images in label_to_images.items():
        indices = np.random.choice(len(images), n_per_class, replace=False)
        images_sampled.extend([images[i] for i in indices])
        labels_sampled.extend([label] * n_per_class)

    return torch.stack(images_sampled), torch.tensor(labels_sampled)


def sample_with_half_worst_classes(
    n_samples,
    model,
    data_loader,
    batch_size=128,
    transformations=[],
    seed=0,
    loss_fun=F.mse_loss,
    device="cpu",
):
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

    worst_classes = sorted(losses_by_class, key=lambda x: np.mean(losses_by_class[x]))[
        : len(losses_by_class) // 2
    ]

    label_to_images = {k: v for k, v in label_to_images.items() if k in worst_classes}
    n_per_class = n_samples // len(label_to_images)
    images_sampled = []
    labels_sampled = []
    np.random.seed(seed)
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
