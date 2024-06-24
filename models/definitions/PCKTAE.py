import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
import torch.optim as optim

from models.definitions.base_model import LightningBaseModel


class PocketAutoencoder(LightningBaseModel):
    '''
    this the autoencoder of Federico. It is small and has flexible hidden dimensions and number of channels.'''

    def __init__(self, hidden_dim=10, n_input_channels=1, input_size=28, path=None):

        if path is not None:
            hidden_dim, n_input_channels, input_size = self.get_n_channels_and_hidden_dim(path)

        super().__init__()
        self.hidden_dim = hidden_dim
        self.n_input_channels = n_input_channels
        self.input_size = input_size
        self.encoder = nn.Sequential(
            nn.Conv2d(n_input_channels, n_input_channels * 3, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm2d(n_input_channels * 3),
            nn.Dropout(0.2),
            nn.GELU(),
            nn.Conv2d(n_input_channels * 3, n_input_channels * 9, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm2d(n_input_channels * 9),
            nn.Dropout(0.2),
            nn.GELU()
        )
        # Calculate the output size after encoder
        self.encoder_out_size = self._get_encoder_out_size()
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(n_input_channels * 9, n_input_channels * 3, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(n_input_channels * 3),
            nn.Dropout(0.2),
            nn.GELU(),
            nn.ConvTranspose2d(n_input_channels * 3, n_input_channels, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(n_input_channels),
            nn.Dropout(0.2),
            nn.ReLU()
        )
        # Define linear layers
        self.encoder_out_linear = nn.Linear(self.encoder_out_size, hidden_dim)
        self.decoder_in_linear = nn.Linear(hidden_dim, self.encoder_out_size)
        # Initialize weights
        self._init_weights_xavier()

    def _get_encoder_out_size(self):
        # Test with dummy input to determine output size of encoder
        test_input = torch.randn(1, self.n_input_channels, self.input_size, self.input_size)
        encoder_output = self.encoder(test_input)
        return encoder_output.view(encoder_output.size(0), -1).shape[1]

    def encode(self, x):
        x = self.encoder(x)
        x = x.view(x.size(0), -1)
        x = self.encoder_out_linear(x)
        return x

    def decode(self, x):
        x = self.decoder_in_linear(x)
        x = x.view(x.size(0), self.n_input_channels * 9, self.input_size // 4, self.input_size // 4)
        x = self.decoder(x)
        return x

    def forward(self, x):
        x = self.encode(x)
        x = self.decode(x)
        return x

    def _init_weights_xavier(self):
        def init_weights(m):
            if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
                init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                init.xavier_uniform_(m.weight)
                init.constant_(m.bias, 0.01)

        # Apply the initialization to all layers
        self.apply(init_weights)

    def training_step(self, batch):
        x, _ = batch
        x_hat = self.forward(x)
        loss = F.mse_loss(x_hat, x)
        return loss

    def validation_step(self, batch):
        x, _ = batch
        x_hat = self.forward(x)
        loss = F.mse_loss(x_hat, x)
        return loss

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=1e-3)
        return optimizer

    def get_latent_space(self, x):
        x = self.encode(x)
        return x

    def get_n_channels_and_hidden_dim(self, path):
        if 'mnist' in path.lower():
            n_channels = 1
            input_size = 28
        elif 'cifar10' or 'cifar100' in path.lower():
            n_channels = 3
            input_size = 32
        else:
            raise ValueError("Invalid dataset")
        _, _, hidden_dim, _ = path.split('_')
        return int(hidden_dim), int(n_channels), int(input_size)
