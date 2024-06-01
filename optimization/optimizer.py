from pathlib import Path

import numpy as np
import cvxpy as cp
import torch
from lightning import Trainer
from torch.utils.data import DataLoader, TensorDataset

from optimization.base_optimizer import BaseOptimizer
from optimization.mlp import MLP


class AffineFitting(BaseOptimizer):
    def __init__(self, z1, z2, lamda):
        """
        Initializes the AffineFitting class.

        Parameters:
            z1 (np.ndarray): Input data matrix of shape (n_samples, latent_dim1)
            z2 (np.ndarray): Output data matrix of shape (n_samples, latent_dim2)
            lamda (float): Regularization parameter
        """
        super().__init__(z1, z2)
        self.lamda = lamda
        self.A_aff = cp.Variable((self.latent_dim2, self.latent_dim1))
        self.b_aff = cp.Variable(self.latent_dim2)
        self.problem = None

    def define_loss(self):
        """
        Defines the loss function for the affine fitting problem.
        """
        n_samples = self.z1.shape[0]
        residuals = [self.A_aff @ self.z1[i] + self.b_aff - self.z2[i] for i in range(n_samples)]
        loss_aff = cp.norm2(cp.vstack(residuals))**2 + self.lamda * cp.norm(self.A_aff, 'fro')**2
        return loss_aff

    def get_results(self):
        """
        Returns the results of the optimization problem.

        Returns:
            float: Optimal value of the problem
            np.ndarray: Optimal values for A_aff
            np.ndarray: Optimal values for b_aff
        """
        print("Getting results")
        if self.problem is None:
            raise Exception("The problem has not been solved yet.")
        return self.problem.value, self.A_aff.value, self.b_aff.value

    def print_results(self):
        """
        Prints the results of the optimization problem.
        """
        opt_value, A_aff_opt, b_aff_opt = self.get_results()
        print("Optimal value: ", opt_value)
        print(A_aff_opt)
        print(b_aff_opt)

    def save_results(self, path):
        # As numpy arrays
        A = self.A_aff.value
        b = self.b_aff.value
        np.savez(path, A=A, b=b)
        return print("Results saved at ", path)

    def transform(self, z1):
        return z1 @ self.A_aff.value.T + self.b_aff.value

    @classmethod
    def from_file(cls, path):
        """
        Loads the results of the optimization problem from a file.

        Parameters:
            path (str): Path to the file containing the results

        Returns:
            AffineFitting: Instance of the AffineFitting class with the loaded results
        """
        data = np.load(path)
        A = data['A']
        b = data['b']
        latent_dim2, latent_dim1 = A.shape
        instance = cls(np.zeros((1, latent_dim1)), np.zeros((1, latent_dim2)), 0)
        instance.A_aff = cp.Parameter((latent_dim2, latent_dim1), value=A)
        instance.b_aff = cp.Parameter(latent_dim2, value=b)
        return instance


class LinearFitting(BaseOptimizer):
    def __init__(self, z1, z2, lamda):
        """
        Initializes the LinearFitting class.

        Parameters:
            z1 (np.ndarray): Input data matrix of shape (n_samples, 32)
            z2 (np.ndarray): Output data matrix of shape (n_samples, 32)
            lamda (float): Regularization parameter
        """
        super().__init__(z1, z2)
        self.lamda = lamda
        self.A = cp.Variable((self.latent_dim2, self.latent_dim1))
        self.problem = None

    def define_loss(self):
        """
        Defines the loss function for the matrix fitting problem.
        """
        n_samples = self.z1.shape[0]
        residuals = [self.A @ self.z1[i] - self.z2[i] for i in range(n_samples)]
        loss = cp.norm2(cp.vstack(residuals))**2 + self.lamda * cp.norm(self.A, 'fro')**2
        return loss

    def get_results(self):
        """
        Returns the results of the optimization problem.

        Returns:
            float: Optimal value of the problem
            np.ndarray: Optimal values for A
        """
        if self.problem is None:
            raise Exception("The problem has not been solved yet.")
        return self.problem.value, self.A.value

    def print_results(self):
        """
        Prints the results of the optimization problem.
        """
        opt_value, A_opt = self.get_results()
        print("Optimal value: ", opt_value)
        print(A_opt)

    def save_results(self, path):
        A = self.A.value
        np.save(path, A)
        return print("Results saved at ", path)

    def transform(self, z1):
        return z1 @ self.A.value.T

    @classmethod
    def from_file(cls, path):
        """
        Loads the results of the optimization problem from a file.

        Parameters:
            path (str): Path to the file containing the results

        Returns:
            LinearFitting: Instance of the LinearFitting class with the loaded results
        """
        A = np.load(path)
        latent_dim2, latent_dim1 = A.shape
        instance = cls(np.zeros((1, latent_dim1)), np.zeros((1, latent_dim2)), 0)
        instance.A = cp.Parameter((latent_dim2, latent_dim1), value=A)
        return instance


class NeuralNetworkFitting(BaseOptimizer):
    def __init__(self, z1, z2, hidden_dim, lamda, learning_rate=0.01, epochs=100):
        """
        Defines a neural network mapping between two latent spaces. Uses the MLP model.

        Parameters:
            z1 (np.ndarray): Input data matrix of shape (n_samples, latent_dim1)
            z2 (np.ndarray): Output data matrix of shape (n_samples, latent_dim2)
            hidden_dim (int): Number of neurons in the hidden layer
            lamda (float): Regularization parameter
            learning_rate (float): Learning rate for the optimizer
            epochs (int): Number of epochs to train the network
        """
        super().__init__(z1, z2)
        self.epochs = epochs
        self.model = MLP(self.latent_dim1, hidden_dim, self.latent_dim2, learning_rate, lamda)

        self.z1 = torch.tensor(z1, dtype=torch.float32)
        self.z2 = torch.tensor(z2, dtype=torch.float32)

    def define_loss(self):
        """
        Defines the loss function for the neural network fitting problem.
        """
        pass

    def fit(self):
        """
        Trains the neural network model.
        """
        trainer = Trainer(max_epochs=self.epochs)
        trainer.fit(self.model, DataLoader(TensorDataset(self.z1, self.z2), batch_size=self.z1.size(0), shuffle=True))

    def get_results(self):
        """
        Returns the results of the optimization problem.

        Returns:
            torch.nn.Module: Trained model
        """
        return self.model

    def save_results(self, path):
        """
        Saves the model parameters to a file.

        Parameters:
            path (str): Path to save the model parameters
        """
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        torch.save(self.model, str(path) + '.pth')
        print("Model saved at ", path)

    def print_results(self):
        """
        Prints the results of the optimization problem.
        """
        print("Model: ", self.model)

    def transform(self, z1):
        """
        Applies the trained model to new data.

        Parameters:
            z1 (np.ndarray): New input data matrix of shape (n_samples, latent_dim1)

        Returns:
            np.ndarray: Transformed data matrix of shape (n_samples, latent_dim2)
        """
        self.model.eval()
        z1 = torch.tensor(z1, dtype=torch.float32)
        with torch.no_grad():
            z1 = self.model(z1).numpy()
        return z1
