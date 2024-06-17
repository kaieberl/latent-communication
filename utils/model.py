import torch
import torchvision.transforms as transforms
import matplotlib.pyplot as plt

device = torch.device('cuda') if torch.cuda.is_available() else 'cpu'


def load_model(model_name, model_path=None, in_channels=1, size=7, latent_size = 0, name_dataset = None, seed = 0, *args, **kwargs):
    """
    Load the model from the given path.
    """
    model_name = model_name.lower()
    if model_name == 'vae':
        from models.definitions.vae import VAE
        model = VAE(in_dim=784, dims=[256, 128, 64, 32], distribution_dim=16)
    elif model_name == 'resnet':
        from models.definitions.resnet import ResNet
        model = ResNet()
    elif model_name == 'vit':
        from models.definitions.vit import ViT
        model = ViT()
    elif model_name == 'ae':
        from models.definitions.ae import LightningAutoencoder as AE
        model = AE()
    elif model_name == 'resnet_ae':
        from models.definitions.resnet_ae import ResnetAE
        model = ResnetAE(in_channels, 50, size)
    elif model_name == 'resnet_vae':
        from models.definitions.resnet_vae import ResnetVAE
        model = ResnetVAE(50, in_channels)
    elif model_name == 'pcktae':
        from models.definitions.smallae import PocketAutoencoder
        model = PocketAutoencoder(path=model_path.split("/")[-1])
    elif model_name == "verysmall-ae":
        from models.definitions.ae_latentdim10 import VerySmallAutoencoder
        model = VerySmallAutoencoder()
    else:
        raise ValueError(f"Unknown model name: {model_name}")

    if model_path is not None:
        model.load_state_dict(torch.load(model_path, map_location=device))
    return model


def load_models(cfg):
    if cfg.dataset == 'mnist':
        in_channels = 1
        size = 7
    elif cfg.dataset == 'cifar10':
        in_channels = 3
        size = 8
    model1 = load_model(cfg.model1.name, cfg.model1.path, in_channels, size)
    model2 = load_model(cfg.model2.name, cfg.model2.path, in_channels, size)
    return model1, model2


def get_transformations(model_name):
    """
    Get the corresponding data transformations for the given model.
    """
    model_name = model_name.lower()
    if model_name == 'vae':
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
    elif model_name == 'vit':
        return [
            transforms.ToTensor(),
            transforms.Resize((224, 224)),
            transforms.Lambda(lambda x: x.repeat(3, 1, 1))
        ]
    elif model_name in ['ae', 'resnet_ae', 'resnet_vae', 'pcktae','verysmall-ae']:
        return [
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
        ]
    else:
        raise ValueError(f"Unknown model name: {model_name}")


def transformations(cfg):
    transformations1 = get_transformations(cfg.model1.name)
    transformations2 = get_transformations(cfg.model2.name)
    return transformations1, transformations2


def get_accuracy(model, test_loader):
    """
    Calculate the accuracy of the model on the given test set.
    """
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for data in test_loader:
            images, labels = data
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    # Calculate accuracy
    accuracy = 100 * correct / total
    return accuracy


def get_reconstruction_error(model, test_loader):
    """
    Calculate the reconstruction error of the model on the given test set.
    """
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for data in test_loader:
            images, _ = data
            images = images.to(device)
            outputs = model(images)
            loss = model.loss_function(images, *outputs) if isinstance(outputs, tuple) else model.loss_function(images,
                                                                                                                outputs)
            total_loss += loss.item()
            # plot the image
            # plt.imshow(images[0].cpu().numpy().reshape(28, 28), cmap='gray')
            # plt.axis('off')
            # plt.show()
            # if isinstance(outputs, tuple):
            #     outputs = outputs[0]
            # plt.imshow(outputs[0].cpu().numpy().reshape(28, 28), cmap='gray')
            # plt.axis('off')
            # plt.show()
            # # plot pixel-wise difference
            # plt.imshow((images[0] - outputs[0]).cpu().numpy().reshape(28, 28), cmap='gray')
            # plt.axis('off')
            # plt.show()
    return total_loss / len(test_loader.dataset)