import lightning as L
import torch
import torch.utils.data
from lightning import Trainer, LightningModule
from torch import optim
from torch.nn import functional as F
from torchvision import models
from torchvision.datasets import MNIST
from torchvision.transforms import transforms


class MNISTDataModule(L.LightningDataModule):
    def __init__(self, data_dir=".", batch_size=16):
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


class MNISTClassifier(LightningModule):
    def __init__(self):
        super().__init__()
        self.model = models.vit_b_16(weights=None, num_classes=10)

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
        optimizer = optim.Adam(self.parameters(), lr=1e-3)
        return optimizer


# Define your seeds
seeds = [0, 1, 2, 3, 4]

# Train the model for each seed
for seed in seeds:
    # set the seed
    L.seed_everything(seed)

    # Train the model
    data_module = MNISTDataModule()
    model = MNISTClassifier()
    trainer = Trainer(max_epochs=10)
    trainer.fit(model, datamodule=data_module)

    # Save the model
    torch.save(model.state_dict(), f'vit_mnist_seed{seed}.pth')
