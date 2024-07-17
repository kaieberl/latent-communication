"""Defines a multi-layer perceptron used for mapping."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from lightning import LightningModule


class NoisyReLU(nn.Module):
    def __init__(self, sigma=0.1):
        super(NoisyReLU, self).__init__()
        self.sigma = sigma

    def forward(self, x):
        noise = torch.randn_like(x) * self.sigma
        return F.relu(x + noise)


class MLP(LightningModule):
    def __init__(self, source_dim, hidden_dim, target_dim, learning_rate, lamda, dropout=0, noise_sigma=0) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.learning_rate = learning_rate
        self.lamda = lamda
        self.dropout = dropout
        self.noise_sigma = noise_sigma
        if noise_sigma > 0:
            activation = NoisyReLU(noise_sigma)
        else:
            activation = nn.GELU()
        self.model = nn.Sequential(
            nn.Linear(source_dim, hidden_dim),
            activation,
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            activation,
            nn.Dropout(dropout + 0.1 if dropout > 0 else 0),
            nn.Linear(hidden_dim, target_dim)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.to(self.device)
        return self.model(x)

    def configure_optimizers(self):
        return torch.optim.Adam(self.model.parameters(), lr=self.learning_rate, weight_decay=self.lamda)

    def training_step(self, batch, batch_idx):
        source_data, target_data = batch
        loss = F.mse_loss(self.model(source_data), target_data)
        self.log('train_loss', loss)
        return loss

    def validation_step(self, batch, batch_idx):
        source_data, target_data = batch
        loss = F.mse_loss(self.model(source_data), target_data)
        self.log('val_loss', loss)
        return loss


class End2EndMLP(LightningModule):
    def __init__(self, model1, model2, hidden_dim, lamda, learning_rate, epochs, train_loader, val_loader=None):
        """
        Defines a neural network mapping between two latent spaces using an MLP.
        """
        super().__init__()
        self.epochs = epochs
        self.mapping = MLP(model1.hidden_dim, hidden_dim, model2.hidden_dim, learning_rate, lamda)

        self.train_loader = train_loader
        self.val_loader = val_loader
        self.model1 = model1
        self.model2 = model2

    def forward(self, x):
        """Encodes the image using model1, maps it to the target latent space using the MLP, and decodes it using
        model2.
        """
        z1 = self.model1.encode(x)
        z2 = self.mapping(z1)
        x = self.model2.decode(z2)
        return x

    def training_step(self, batch, batch_idx):
        source_data, target_data = batch
        loss = F.mse_loss(self.model(source_data), target_data)
        self.log('train_loss', loss)
        return loss

    def validation_step(self, batch, batch_idx):
        source_data, target_data = batch
        loss = F.mse_loss(self.model(source_data), target_data)
        self.log('val_loss', loss)
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.mapping.parameters(), lr=self.learning_rate, weight_decay=self.lamda)