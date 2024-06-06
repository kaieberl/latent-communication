import torch
from models.definitions.base_model import BaseModel

import torch.nn.init as init
import torch.nn as nn
import torch.nn.functional as F


class PocketAutoencoder(BaseModel):
    def __init__(self):
        super().__init__()
        self.hidden_dim = 49
        self.autoencoder = nn.Module()
        self.autoencoder.encoder = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv2d(
                        1, 3, kernel_size=5, stride=2, padding=2
                    ),
                    nn.BatchNorm2d(3),
                    nn.GELU(),
                    nn.Conv2d(
                        3, 9, kernel_size=5, stride=2, padding=2
                    ),
                    nn.BatchNorm2d(9),
                    nn.Dropout(0.2),
                    nn.GELU(),
                )
            ]
        )
        self.autoencoder.decoder = nn.ModuleList(
            [
                nn.Sequential(
                    nn.ConvTranspose2d(
                        9, 3, kernel_size=4, stride=2, padding=1
                    ),
                    nn.BatchNorm2d(3),
                    nn.GELU(),
                    nn.ConvTranspose2d(
                        3, 1, kernel_size=4, stride=2, padding=1
                    ),
                    nn.BatchNorm2d(
                        1
                    ),  # This should be 1 since the output channel of ConvTranspose2d is 1
                    nn.ReLU(),
                )
            ]
        )
        self.autoencoder.encoder_out = nn.Sequential(
            nn.Linear(
                7 * 7 * 9, self.hidden_dim
            ),  # Adjust based on output of conv layers
            nn.GELU(),
        )
        self.autoencoder.decoder_in = nn.Sequential(
            nn.Linear(self.hidden_dim, 7 * 7 * 9),
            nn.GELU(),
        )
        

    def encode(self, x):
        for layer in self.autoencoder.encoder:
            x = layer(x)
        x = x.view(x.size(0), -1)
        x = self.autoencoder.encoder_out(x)
        return x

    def decode(self, x):
        x = self.autoencoder.decoder_in(x)
        x = x.view(x.size(0), 9, 7, 7)
        for layer in self.autoencoder.decoder:
            x = layer(x)
        return x

    def forward(self, x):
        for layer in self.autoencoder.encoder:
            x = layer(x)
        x = torch.flatten(x, 1)
        x = self.autoencoder.encoder_out(x)
        x = self.autoencoder.decoder_in(x)
        x = x.view(-1, 9, 7, 7)
        for layer in self.autoencoder.decoder:
            x = layer(x)
        return x

    def init_weights(self, m):
        if type(m) == nn.Linear:
            init.kaiming_normal_(m.weight)
            m.bias.data.fill_(0.01)

    def training_step(self, batch):
        x = batch
        x_hat = self.forward(x)
        loss = F.mse_loss(x_hat, x)
        return loss

    def validation_step(self, batch):
        x = batch
        x_hat = self.forward(x)
        loss = F.mse_loss(x_hat, x)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer

    def get_latent_space(self, x):
        for layer in self.autoencoder.encoder:
            x = layer(x)
        x = x.view(x.size(0), -1)
        x = self.autoencoder.encoder_out(x)
        return x


class AEClassifier(BaseModel):
    def __init__(self, n_classes):
        super().__init__()
        self.size_classes = n_classes
        self.autoencoder = nn.Module()
        self.autoencoder.encoder = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv2d(1, 32, kernel_size=3, stride=2, padding=1),
                    nn.BatchNorm2d(32),
                    nn.GELU(),
                ),
                nn.Sequential(
                    nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
                    nn.BatchNorm2d(64),
                    nn.GELU(),
                ),
                nn.Sequential(
                    nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
                    nn.BatchNorm2d(128),
                    nn.GELU(),
                ),
                nn.Sequential(
                    nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),
                    nn.BatchNorm2d(256),
                    nn.GELU(),
                ),
            ]
        )
        self.autoencoder.encoder_out = nn.Sequential(
            nn.Linear(1024, 500),
            nn.GELU(),
        )
        self.autoencoder.classifier = nn.ModuleList(
            [
                nn.Sequential(
                    nn.LazyLinear(100),
                    nn.ReLU(),
                    nn.Linear(100, self.size_classes),
                    nn.GELU(),
                )
            ]
        )

    def encode(self, x):
        for layer in self.autoencoder.encoder:
            x = layer(x)
        x = x.view(x.size(0), -1)
        x = self.autoencoder.encoder_out(x)
        return x

    def decode(self, x):
        return x

    def forward(self, x):
        for layer in self.autoencoder.encoder:
            x = layer(x)
        x = x.view(x.size(0), -1)
        x = self.autoencoder.encoder_out(x)
        print("wah")
        for layer in self.autoencoder.classifier:
            x = layer(x)
        return x

    def init_weights(self, m):
        if type(m) == nn.Linear:
            init.kaiming_normal_(m.weight)
            m.bias.data.fill_(0.01)

    def training_step(self, batch, prediction):
        x = batch
        x_hat = self.forward(x)
        loss = F.mse_loss(x_hat, prediction)
        self.log("train_loss", loss)
        return loss

    def test_step(self, batch, prediction):
        x = batch
        x_hat = self.forward(x)
        loss = F.mse_loss(x_hat, prediction)
        self.log("val_loss", loss)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer

    def get_latent_space(self, x):
        for layer in self.autoencoder.encoder:
            x = layer(x)
        x = x.view(x.size(0), -1)
        x = self.autoencoder.encoder_out(x)
        return x
