import torch
import torchvision.transforms as transforms

device = torch.device('cuda') if torch.cuda.is_available() else 'cpu'


def load_model(model_name, model_path=None, in_channels=1, size=7, latent_size=8, name_dataset=None, seed=0, *args,
               **kwargs):
    """
    Load the model from the given path.
    """
    model_name = model_name.lower()
    if model_name == 'vae':
        from models.definitions.vae import VAE
        model = VAE(in_dim=784, latent_dim=latent_size, return_var=True)
    elif model_name == 'resnet':
        from models.definitions.resnet import ResNet
        model = ResNet()
    elif model_name == 'vit':
        from models.definitions.vit import ViT
        model = ViT()
    elif model_name == 'ae':
        from models.definitions.ae import LightningAutoencoder as AE
        model = AE()
    elif model_name == 'pcktae':
        from models.definitions.PCKTAE import PocketAutoencoder
        if model_path is None:
            model = PocketAutoencoder(latent_size)
        else:
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
    elif cfg.dataset == 'fmnist':
        in_channels = 1
        size = 7
    elif cfg.dataset == 'cifar10':
        in_channels = 3
        size = 8
    else:
        raise ValueError(f"Unknown dataset: {cfg.dataset}")
    model1 = load_model(cfg.model1.name, cfg.model1.path, in_channels, size, cfg.model1.latent_size)
    model2 = load_model(cfg.model2.name, cfg.model2.path, in_channels, size, cfg.model2.latent_size)
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
    elif model_name in ['ae', 'pcktae', 'verysmall-ae']:
        return [
            transforms.ToTensor()
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
            loss = model.reconstruction_loss(images, *outputs) if isinstance(outputs, tuple) else model.reconstruction_loss(images,
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
