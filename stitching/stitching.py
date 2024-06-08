from pathlib import Path

import numpy as np
import torch
from omegaconf import DictConfig
import hydra

from utils.dataloaders.dataloader_mnist_single import DataLoaderMNIST
from utils.model import load_models, get_accuracy, get_transformations, get_reconstruction_error

device = torch.device('cuda') if torch.cuda.is_available() else 'mps' if torch.has_mps else 'cpu'


def get_stitched_output(model1, model2, mapping, images):
    latent_space1 = model1.get_latent_space(images).to(dtype=torch.float32)
    latent_space_stitched = mapping.transform(latent_space1.detach().cpu())
    #Convert to tensor if necessary and to the right dtype
    if isinstance(latent_space_stitched, np.ndarray):
        latent_space_stitched = torch.tensor(latent_space_stitched, dtype=torch.float32)
    elif isinstance(latent_space_stitched, torch.Tensor):
        latent_space_stitched = latent_space_stitched.to(dtype=torch.float32)
    
    outputs = model2.decode(latent_space_stitched.to(device))
    return outputs


def load_mapping(cfg):
    name = f'{cfg.mapping}_{cfg.model1.name}_{cfg.model1.seed}_{cfg.model2.name}_{cfg.model2.seed}_{cfg.num_samples}'
    path = Path(cfg.base_dir) / 'results/transformations' / cfg.storage_path / name
    if cfg.mapping == 'Linear':
        from optimization.optimizer import LinearFitting
        mapping = LinearFitting.from_file(path)
    elif cfg.mapping == 'Affine':
        from optimization.optimizer import AffineFitting
        mapping = AffineFitting.from_file(path)
    elif cfg.mapping == 'NeuralNetwork':
        from optimization.optimizer import NeuralNetworkFitting
        mapping = NeuralNetworkFitting.from_file(path)
    else:
        raise ValueError("Invalid experiment name")
    return mapping


@hydra.main(version_base="1.1", config_path="../config")
def main(cfg: DictConfig) -> None:
    cfg.base_dir = Path(hydra.utils.get_original_cwd()).parent
    print("Using device ", device)
    model1, model2 = load_models(cfg)

    # Initialize data loader
    # TODO: this uses the wrong data loader for model 2 if it has different transformations, e.g. vit and resnet
    data_loader_model = DataLoaderMNIST(128, get_transformations(cfg.model1.name), seed=10, base_path = cfg.base_dir)
    test_loader = data_loader_model.get_test_loader()

    if cfg.model1.name in ['ae', 'vae', 'resnet_ae', 'resnet_vae']:
        # print reconstruction error
        reconstruction_error1 = get_reconstruction_error(model1, test_loader)
        reconstruction_error2 = get_reconstruction_error(model2, test_loader)
        print(f'Reconstruction error of {cfg.model1.name} {cfg.model1.seed} on the test images: %.2f' % reconstruction_error1)
        print(f'Reconstruction error of {cfg.model2.name} {cfg.model2.seed} on the test images: %.2f' % reconstruction_error2)
    else:
        # Print accuracy for model 1 on test set
        accuracy1 = get_accuracy(model1, test_loader)
        accuracy2 = get_accuracy(model2, test_loader)
        print(f'Accuracy of {cfg.model1.name} {cfg.model1.seed} on the test images: %.2f %%' % accuracy1)
        print(f'Accuracy of {cfg.model2.name} {cfg.model2.seed} on the test images: %.2f %%' % accuracy2)

    mapping = load_mapping(cfg)

    # Print accuracy of stitched model
    correct = 0
    total = 0
    with torch.no_grad():
        for data in test_loader:
            images, labels = data
            images, labels = images.to(device), labels.to(device)
            outputs = get_stitched_output(model1, model2, mapping, images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy_stitched = 100 * correct / total
    print('Accuracy of stitched model on the test images: %.2f %%' % accuracy_stitched)


if __name__ == '__main__':
    main()
