# Import relevant libraries
import os

import numpy as np
import torch
import torchvision.transforms as transforms

from utils.dataloader_mnist_single import DataLoaderMNIST

# Configuration
seed1 = 1
seed2 = 2
root_folder = os.path.dirname(os.getcwd())  # folder for latent-communcation
# 12, 13, 14, 23, 24 ,34
config = {
    'path1': root_folder+f"/models/checkpoints/ResNet/MNIST/model_seed{seed1}.pth",
    'modelname1': 'resnet',
    'seed1': f'{seed1}',
    'path2': root_folder+f"/models/checkpoints/ResNet/MNIST/model_seed{seed2}.pth",
    'modelname2': 'resnet',
    'seed2': f'{seed2}',
    'num_samples': '100',
    'storage_path': 'ResNet-LinearTransform'

}


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


def load_models():
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


def get_stitched_output(model1,model2,A,images):
    latent_space1 = model1.get_latent_space(images)
    latent_space_stitched = []
    for i in range(latent_space1.shape[0]):
        latent_space_stitched.append(A @ latent_space1[i,:])
    latent_space_stitched = torch.stack(latent_space_stitched)
    outputs = model2.decode(latent_space_stitched)
    return outputs


def main():
    model1, model2 = load_models()

    # Initialize data loader
    data_loader_model = DataLoaderMNIST(128, get_transformations(config['modelname1']), seed=10)
    test_loader = data_loader_model.get_test_loader()

    # Print accuracy for model 1 on test set
    accuracy1 = get_accuracy(model1, test_loader)
    accuracy2 = get_accuracy(model2, test_loader)
    print(f'Accuracy of {config["modelname1"]} {seed1} on the test images: %.2f %%' % accuracy1)
    print(f'Accuracy of {config["modelname1"]} {seed2} on the test images: %.2f %%' % accuracy2)

    #Get the transformation
    name = 'Linear_' + config['modelname1'] + '_' + config['seed1'] + '_' + config['modelname2'] + '_' + config['seed2'] + '_' + config['num_samples'] + '.npy'
    path = root_folder + '/results/transformations/' + config['storage_path'] + '/' + name
    A = np.load(path)
    A = torch.tensor(A).float()

    #Print accuracy of stitched model
    correct = 0
    total = 0
    with torch.no_grad():
        for data in test_loader:
            images, labels = data
            outputs = get_stitched_output(model1,model2,A,images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy_stitched = 100 * correct / total
    print('Accuracy of stitched model on the test images: %.2f %%' % accuracy_stitched)


if __name__ == '__main__':
    main()