import lightning as L
import torch

from resnet.utils import visualize_latent_space_pca, visualize_latent_space_tsne
from vit.train_vit import MNISTDataModule, MNISTClassifier


def setup_model(seed, device):
    L.seed_everything(seed)
    model = MNISTClassifier().to(device)
    model.load_state_dict(torch.load(f"vit/models/vit_mnist_seed{seed}_new.pth"))
    model.eval()
    return model


def save_latent_space(model, dataloader, prefix, seed):
    latents, labels = model.get_latent_space(dataloader)
    torch.save(labels, f"vit/models/labels_{prefix}.pt")
    torch.save(latents, f"vit/models/latent_space_vit_seed{seed}_{prefix}.pt")


def load_and_visualize_latent_space(seed):
    labels = torch.load(f"vit/models/labels_test.pt", map_location='cpu')
    latents = torch.load(f"vit/models/latent_space_vit_seed{seed}_test.pt", map_location='cpu')
    visualize_latent_space_pca(latents, labels, f"vit/figures/latent_space_pca_seed{seed}.png")
    visualize_latent_space_tsne(latents, labels, 10, f"vit/figures/latent_space_tsne_seed{seed}.png")


def main():
    seed = 0
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data_module = MNISTDataModule(data_dir="vit", batch_size=32)

    model = setup_model(seed, device)

    # For validation set
    save_latent_space(model, data_module.val_dataloader(), "test", seed)
    load_and_visualize_latent_space(seed)

    # For training set
    save_latent_space(model, data_module.train_dataloader(), "train", seed)


if __name__ == "__main__":
    main()
