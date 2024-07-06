import torch.utils.data
from lightning import LightningModule
from abc import ABC, abstractmethod

import torch.nn as nn
from abc import abstractmethod

from tqdm import tqdm


class LightningBaseModel(LightningModule, ABC):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @abstractmethod
    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """
        Takes the latent code as input and returns the reconstructed image.
        """
        pass

    @abstractmethod
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """
        Encodes the input by passing through the encoder network
        """
        pass

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Passes through the encoder and decoder and returns the output.
        Using reparameterization trick to sample from the latent space.
        """
        z = self.encode(x)
        return self.decode(z)

    @abstractmethod
    def get_latent_space(self, x: torch.Tensor) -> torch.Tensor:
        """
        Returns the latent space representation of the input. Last Layer of the Encoder before the mean and variance.
        """
        pass

    @torch.no_grad()
    def get_latent_space_from_dataloader(self, dataloader):
        """
        Returns the latent space representation of the input.
        """
        latents = []
        labels = []
        self.eval()
        for x, y in tqdm(dataloader):
            x = x.to(self.device)
            labels.append(y)
            latents.append(self.get_latent_space(x).detach().cpu())
        return torch.cat(latents), torch.cat(labels)

    @abstractmethod
    def configure_optimizers(self):
        """
        Define the optimizer for training.
        Override this method to specify your optimizer configuration.
        """
        pass

    def reconstruction_loss(self, x: torch.Tensor, x_hat: torch.Tensor) -> torch.Tensor:
        """
        Calculates the reconstruction loss.
        """
        return nn.functional.mse_loss(x_hat, x)


class BaseModel(nn.Module, ABC):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.device = torch.device('cuda') if torch.cuda.is_available() else 'cpu'

    @abstractmethod
    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """
        Takes the latent code as input and returns the reconstructed image.
        """
        pass

    @abstractmethod    
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """"
        Encodes the input by passing through the encoder network
        """
        pass

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Passes through the encoder and decoder and returns the output.
        Using reparameterization trick to sample from the latent space.
        """
        z = self.encode(x)
        return self.decode(z)

    @abstractmethod
    def get_latent_space(self, x: torch.Tensor) -> torch.Tensor:
        """
        Returns the latent space representation of the input. Last Layer of the Encoder before the mean and variance.
        """
        pass

    @torch.no_grad()
    def get_latent_space_from_dataloader(self, dataloader):
        """
        Returns the latent space representation of the input.
        """
        latents = []
        labels = []
        self.eval()
        for x, y in tqdm(dataloader):
            x = x.to(self.device)
            labels.append(y)
            latents.append(self.get_latent_space(x).detach().cpu())
        return torch.cat(latents), torch.cat(labels)