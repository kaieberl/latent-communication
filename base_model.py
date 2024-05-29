import torch.utils.data
from lightning import LightningModule
from abc import ABC, abstractmethod

import torch.nn as nn
from abc import abstractmethod


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

    @abstractmethod
    def configure_optimizers(self):
        """
        Define the optimizer for training.
        Override this method to specify your optimizer configuration.
        """
        pass


class BaseModel(nn.Module, ABC):

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

    def get_latent_space_from_dataloader(self, x: torch.utils.data.DataLoader) -> torch.Tensor:
        """
        Returns the latent space representation of the input. Last Layer of the Encoder before the mean and variance.
        """
        pass