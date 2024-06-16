import torch
import torch.nn as nn
import torch.nn.functional as F

from models.definitions.base_model import LightningBaseModel


class ResNetBlock(nn.Module):
    def __init__(self, in_channels):
        super(ResNetBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(in_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(in_channels)

    def forward(self, x):
        identity = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out += identity
        out = self.relu(out)
        return out


class Encoder(nn.Module):
    def __init__(self, in_channels, hidden_dim, size):
        super(Encoder, self).__init__()
        self.initial = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )
        self.resblock = ResNetBlock(64)
        self.fc = nn.Linear(64 * size * size, hidden_dim)

    def forward(self, x):
        x = self.initial(x)
        x = self.resblock(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


class Decoder(nn.Module):
    def __init__(self, hidden_dim, out_channels, size):
        super(Decoder, self).__init__()
        self.fc = nn.Linear(hidden_dim, 64 * size * size)
        self.resblock = ResNetBlock(64)
        self.final = nn.Sequential(
            nn.ConvTranspose2d(64, 64, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(64, out_channels, kernel_size=4, stride=2, padding=1),
            nn.Tanh()
        )
        self.size = size

    def forward(self, x):
        x = self.fc(x)
        x = x.view(x.size(0), 64, self.size, self.size)
        x = self.resblock(x)
        x = self.final(x)
        return x


class ResnetAE(LightningBaseModel):
    def __init__(self, in_channels, hidden_dim, size, *args, **kwargs):
        super().__init__()
        self.encoder = Encoder(in_channels, hidden_dim, size)
        self.decoder = Decoder(hidden_dim, in_channels, size)
        self.hidden_dim = hidden_dim

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

    def training_step(self, batch, batch_idx):
        x, _ = batch
        x_hat = self(x)
        loss = F.mse_loss(x_hat, x)
        self.log('train_loss', loss)
        return loss

    def validation_step(self, batch, batch_idx):
        x, _ = batch
        x_hat = self(x)
        loss = F.mse_loss(x_hat, x)
        self.log('val_loss', loss)
        return loss

    def loss_function(self, x_hat, x):
        l1_norm = sum(p.abs().sum() for p in self.parameters())
        return F.mse_loss(x_hat, x) + 1e-3 * l1_norm

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=1e-3)

    def encode(self, x):
        return self.encoder(x)

    def decode(self, z):
        return self.decoder(z)

    def get_latent_space(self, x):
        return self.encode(x)

    def get_latent_space_from_dataloader(self, dataloader):
        latents = []
        labels = []
        for x, y in dataloader:
            latents.append(self.get_latent_space(x))
            labels.append(y)
        return torch.cat(latents), torch.cat(labels)
