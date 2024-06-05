import torch
from models.definitions.base_model import BaseModel

import torch.nn.init as init
import torch.nn as nn
import torch.nn.functional as F



class PocketAutoencoder(BaseModel):
    def __init__(self):
        super().__init__()
        self.autoencoder = nn.Module()
        self.autoencoder.encoder = nn.ModuleList([
        nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(16),
            nn.GELU(),),
        nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.GELU()),
        ])
        self.autoencoder.decoder = nn.ModuleList([
            nn.Sequential(
                nn.ConvTranspose2d(32, 16, kernel_size=3, stride=2, padding=1),
                nn.BatchNorm2d(16),
                nn.GELU()),
            nn.Sequential(
                nn.ConvTranspose2d(16, 1, kernel_size=3, stride=2, padding=1),
                nn.BatchNorm2d(64),
                nn.ReLU()),
        ])
        self.autoencoder.encoder_out = nn.Sequential(
            nn.LazyLinear(1568, 128),
            nn.GELU(),
        )
        self.autoencoder.decoder_in = nn.Sequential(
            nn.Linear(128, 1568),
            nn.GELU(),
        )
        self.hidden_dim = 128

    def encode(self, x):
        for layer in self.autoencoder.encoder:
            x = layer(x)
        x = x.view(x.size(0), -1)
        x = self.autoencoder.encoder_out(x)
        return x

    def decode(self, x):
        x = self.autoencoder.decoder_in(x)
        x = x.view(x.size(0), 32, 7, 7)
        for layer in self.autoencoder.decoder:
            x = layer(x)
        return x

    def forward(self, x):
        for layer in self.autoencoder.encoder:
            x = layer(x)
        x = x.view(x.size(0), -1)
        x = self.autoencoder.encoder_out(x)
        x = self.autoencoder.decoder_in(x)
        x = x.view(x.size(0), 256, 2, 2)
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
        self.log('train_loss', loss)
        return loss

    def validation_step(self, batch):
        x = batch
        x_hat = self.forward(x)
        loss = F.mse_loss(x_hat, x)
        self.log('val_loss', loss)
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
        self.autoencoder.encoder = nn.ModuleList([
        nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.GELU(),),
        nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.GELU()),
        nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.GELU()),
        nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.GELU(),
        )])
        self.autoencoder.encoder_out = nn.Sequential(
            nn.Linear(1024, 500),
            nn.GELU(),
        )
        self.autoencoder.classifier = nn.ModuleList([
            nn.Sequential(
                nn.LazyLinear(100),
                nn.ReLU(),
                nn.Linear(100, self.size_classes),
                nn.GELU())])

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
        print('wah')
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
        self.log('train_loss', loss)
        return loss

    def test_step(self, batch, prediction):
        x = batch
        x_hat = self.forward(x)
        loss = F.mse_loss(x_hat, prediction)
        self.log('val_loss', loss)
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