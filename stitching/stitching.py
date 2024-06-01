from pathlib import Path

import numpy as np
import torch
import torchvision.transforms as transforms
from omegaconf import DictConfig
import hydra

from utils.dataloader_mnist_single import DataLoaderMNIST


def load_model(model_name, model_path):
    device = torch.device('cuda') if torch.cuda.is_available() else 'cpu'
    
    if model_name == 'vae':
        from models.definitions.vae import VAE
        model = VAE(in_dim=784, dims=[256, 128, 64, 32], distribution_dim=16).to(device)
    elif model_name == 'resnet':
        from models.definitions.resnet import ResNet
        model = ResNet().to(device)
    else:
        raise ValueError(f"Unknown model name: {model_name}")
    
    model.load_state_dict(torch.load(model_path, map_location=device))
    return model


def load_models(cfg):
    model1 = load_model(cfg.model1.name, cfg.model1.path)
    model2 = load_model(cfg.model2.name, cfg.model2.path)
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


def transformations(cfg):
    transformations1 = get_transformations(cfg.model1.name)
    transformations2 = get_transformations(cfg.model2.name)
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


def get_stitched_output(model1, model2, A, images):
    latent_space1 = model1.get_latent_space(images)
    latent_space_stitched = []
    for i in range(latent_space1.shape[0]):
        latent_space_stitched.append(A @ latent_space1[i, :])
    latent_space_stitched = torch.stack(latent_space_stitched)
    outputs = model2.decode(latent_space_stitched)
    return outputs


@hydra.main(config_path="../config", config_name="config_resnet")
def main(cfg: DictConfig) -> None:
    cfg.base_dir = Path(hydra.utils.get_original_cwd()).parent
    model1, model2 = load_models(cfg)

    # Initialize data loader
    data_loader_model = DataLoaderMNIST(128, get_transformations(cfg.model1.name), seed=10)
    test_loader = data_loader_model.get_test_loader()

    # Print accuracy for model 1 on test set
    accuracy1 = get_accuracy(model1, test_loader)
    accuracy2 = get_accuracy(model2, test_loader)
    print(f'Accuracy of {cfg.model1.name} {cfg.model1.seed} on the test images: %.2f %%' % accuracy1)
    print(f'Accuracy of {cfg.model2.name} {cfg.model2.seed} on the test images: %.2f %%' % accuracy2)

    # Get the transformation
    name = f'{cfg.mapping}_{cfg.model1.name}_{cfg.model1.seed}_{cfg.model2.name}_{cfg.model2.seed}_{cfg.num_samples}.npy'
    path = Path(cfg.base_dir) / 'results/transformations' / cfg.storage_path / name
    A = np.load(path)
    A = torch.tensor(A).float()

    # Print accuracy of stitched model
    correct = 0
    total = 0
    with torch.no_grad():
        for data in test_loader:
            images, labels = data
            outputs = get_stitched_output(model1, model2, A, images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy_stitched = 100 * correct / total
    print('Accuracy of stitched model on the test images: %.2f %%' % accuracy_stitched)


if __name__ == '__main__':
    main()
