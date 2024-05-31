import lightning as L
import torch
import torch.utils.data
from lightning import Trainer
from torch import optim
import torch.nn.functional as F
from torchvision import models
from torchvision.datasets import MNIST
from torchvision.transforms import transforms
from tqdm import tqdm

from base_model import LightningBaseModel


class MNISTDataModule(L.LightningDataModule):
    def __init__(self, data_dir=".", batch_size=128):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size

        # Define transformations (same as before)
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize((224, 224)),
            transforms.Lambda(lambda x: x.repeat(3, 1, 1))
        ])

    def setup(self, stage=None):
        MNIST(root=self.data_dir, train=stage == "fit", download=True, transform=self.transform)

    def train_dataloader(self):
        return torch.utils.data.DataLoader(MNIST(root=self.data_dir, train=True, transform=self.transform),
                                           batch_size=self.batch_size, shuffle=True)

    def val_dataloader(self):
        return torch.utils.data.DataLoader(MNIST(root=self.data_dir, train=False, transform=self.transform),
                                           batch_size=self.batch_size, shuffle=False)


class ViT(LightningBaseModel):
    def __init__(self):
        super().__init__()
        self.model = models.vit_b_16(weights=None, num_classes=10)
        # self.model = models.vit_b_16(weights="imagenet21k+imagenet2012", pretrained=True, num_classes=10)

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        images, labels = batch
        outputs = self(images)
        loss = F.cross_entropy(outputs, labels)
        return loss

    def validation_step(self, batch, batch_idx):
        images, labels = batch
        outputs = self(images)
        loss = F.cross_entropy(outputs, labels)
        preds = torch.argmax(outputs, dim=1)
        correct = (preds == labels).sum()
        total = labels.size(0)
        acc = correct / total
        self.log("val_loss", loss, prog_bar=True, logger=True)
        self.log("val_acc", acc, prog_bar=True, logger=True)
        return loss

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=8e-3, weight_decay=0.1)
        lr_scheduler = optim.lr_scheduler.LambdaLR(optimizer, lambda epoch: 1 - epoch / 7)
        return [optimizer], [lr_scheduler]

    def encode(self, x):
        """
        Encodes the input by passing through the encoder network and returns the intermediate layer output.
        """
        # Reshape and permute the input tensor
        x = self.model._process_input(x)
        n = x.shape[0]

        # Expand the class token to the full batch
        batch_class_token = self.model.class_token.expand(n, -1, -1)
        x = torch.cat([batch_class_token, x], dim=1)

        x = self.model.encoder(x)

        # [CLS] token
        return x[:, 0]

    def decode(self, z):
        """
        Takes the latent code as input and returns the reconstructed image.
        """
        return self.model.heads(z)

    def get_latent_space(self, x):
        return self.encode(x)

    @torch.no_grad()
    def get_latent_space_from_dataloader(self, dataloader):
        """
        Returns the latent space representation of the input.
        """
        total_samples = len(dataloader.dataset)
        latent_dim = self.model.hidden_dim
        latents = torch.zeros((total_samples, latent_dim), device=self.device)
        labels = torch.zeros(total_samples, dtype=torch.long, device=self.device)
        dataloader = torch.utils.data.DataLoader(dataloader.dataset, batch_size=dataloader.batch_size, shuffle=False)
        start_idx = 0
        for images, targets in tqdm(dataloader):
            images = images.to(self.device)
            targets = targets.to(self.device)
            latents[start_idx:start_idx + dataloader.batch_size] = self.encode(images)
            labels[start_idx:start_idx + dataloader.batch_size] = targets
            start_idx += dataloader.batch_size
        return latents, labels


if __name__ == '__main__':
    seeds = [0, 1, 2]

    # Train the model for each seed
    for seed in seeds:
        L.seed_everything(seed)

        data_module = MNISTDataModule()
        model = ViT()
        trainer = Trainer(max_epochs=7)
        trainer.fit(model, datamodule=data_module)

        torch.save(model.state_dict(), f'vit_mnist_seed{seed}.pth')
