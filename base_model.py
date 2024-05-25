from lightning import LightningModule
from abc import ABC, abstractmethod

import torch.nn as nn


class LightningBaseModel(LightningModule, ABC):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @abstractmethod
    def decode(self, z):
        """
        Takes the latent code as input and returns the reconstructed image.
        """
        pass

    @abstractmethod
    def encode(self, x):
        """
        Encodes the input by passing through the encoder network
        """
        pass

    def forward(self, x):
        """
        Passes through the encoder and decoder and returns the output.
        Using reparameterization trick to sample from the latent space.
        """
        z = self.encode(x)
        return self.decode(z)

    @abstractmethod
    def get_latent_space(self, x):
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
    def decode(self, z):
        """
        Takes the latent code as input and returns the reconstructed image.
        """
        pass

    @abstractmethod    
    def encode(self, x):
        """"
        Encodes the input by passing through the encoder network
        """
        pass

    def forward(self, x):
        """
        Passes through the encoder and decoder and returns the output.
        Using reparameterization trick to sample from the latent space.
        """
        z = self.encode(x)
        return self.decode(z)

    @abstractmethod
    def get_latent_space(self, x):
        """
        Returns the latent space representation of the input. Last Layer of the Encoder before the mean and variance.
        """
        pass