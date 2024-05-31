"""Trains a linear mapping between two latent spaces, using gradient descent."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from lightning import LightningModule, Trainer


class LinearMapping(LightningModule):
    def __init__(self, source_dim, target_dim) -> None:
        super().__init__()
        self.translation = nn.Linear(
                source_dim, target_dim, device=self.device, dtype=torch.float32, bias=False
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.to(self.device)
        return self.translation(x)

    def configure_optimizers(self):
        return torch.optim.Adam(self.translation.parameters(), lr=1e-3)

    def training_step(self, batch, batch_idx):
        source_data, target_data = batch
        loss = F.mse_loss(self.translation(source_data), target_data)
        self.log('train_loss', loss)
        return loss

    def validation_step(self, batch, batch_idx):
        source_data, target_data = batch
        loss = F.mse_loss(self.translation(source_data), target_data)
        self.log('val_loss', loss)
        return loss


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # load the data
    source_data_train = torch.load("../vit/models/latent_space_vit_seed0_train.pt")
    target_data_train = torch.load("../vit/models/latent_space_vit_seed1_train.pt")

    # sample from trainings data
    torch.manual_seed(0)
    indices = torch.randperm(source_data_train.size(0))[:100]
    source_data_train = source_data_train[indices]

    source_data_test = torch.load("../vit/models/latent_space_vit_seed0_test.pt")
    target_data_test = torch.load("../vit/models/latent_space_vit_seed1_test.pt")

    # Initialize and fit the translator
    translator = LinearMapping(source_data_train.size(1), target_data_train.size(1))
    trainer = Trainer(max_epochs=200)
    trainer.fit(translator, DataLoader(TensorDataset(source_data_train, target_data_train), batch_size=source_data_train.size(0), shuffle=True), DataLoader(TensorDataset(source_data_test, target_data_test), batch_size=source_data_test.size(0)))
    torch.save(translator, "../vit/models/translator_linear.pt")

    # Test the translation and plot the error
    translator.eval()
    translated_data = translator(source_data_test)
    torch.save(translated_data, "../vit/models/latent_space_vit_seed0_test_linear.pt")
