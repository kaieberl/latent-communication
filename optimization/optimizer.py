from pathlib import Path

import numpy as np
import cvxpy as cp
import torch
from torch import nn, optim

from base_optimizer import BaseOptimizer


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


class LinearFitting(BaseOptimizer):
    def __init__(self, z1, z2, lamda):
        """
        Initializes the MatrixFitting class.

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


class NeuralNetworkFitting(BaseOptimizer):
    def __init__(self, z1, z2, hidden_dim, lamda, learning_rate=0.01, epochs=100):
        """
        Initializes the NeuralNetworkFitting class.

        Parameters:
            z1 (np.ndarray): Input data matrix of shape (n_samples, latent_dim1)
            z2 (np.ndarray): Output data matrix of shape (n_samples, latent_dim2)
            hidden_dim (int): Number of neurons in the hidden layer
            lamda (float): Regularization parameter
            learning_rate (float): Learning rate for the optimizer
            epochs (int): Number of epochs to train the network
        """
        super().__init__(z1, z2)
        self.lamda = lamda
        self.learning_rate = learning_rate
        self.epochs = epochs

        self.model = nn.Sequential(
            nn.Linear(self.latent_dim1, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, self.latent_dim2)
        )

        self.criterion = nn.MSELoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate, weight_decay=self.lamda)
        self.z1 = torch.tensor(z1, dtype=torch.float32)
        self.z2 = torch.tensor(z2, dtype=torch.float32)

    def define_loss(self):
        """
        Defines the loss function for the neural network fitting problem.
        """
        def loss_fn():
            predictions = self.model(self.z1)
            loss = self.criterion(predictions, self.z2)
            return loss
        return loss_fn

    def fit(self):
        """
        Fits the neural network to the data.
        """
        self.model.train()
        loss_fn = self.define_loss()
        for epoch in range(self.epochs):
            self.optimizer.zero_grad()
            loss = loss_fn()
            loss.backward()
            self.optimizer.step()
            if (epoch + 1) % 10 == 0:
                print(f"Epoch {epoch + 1}/{self.epochs}, Loss: {loss.item()}")

    def get_results(self):
        """
        Returns the results of the optimization problem.

        Returns:
            float: Final loss value
            nn.Module: Trained model
        """
        self.model.eval()
        with torch.no_grad():
            final_loss = self.define_loss()()
        return final_loss.item(), self.model

    def print_results(self):
        """
        Prints the results of the optimization problem.
        """
        final_loss, model = self.get_results()
        print("Final loss: ", final_loss)
        print("Model parameters: ")
        for param in model.parameters():
            print(param)

    def save_results(self, path):
        """
        Saves the model parameters to a file.

        Parameters:
            path (str): Path to save the model parameters
        """
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        torch.save(self.model.state_dict(), str(path) + '.pth')
        print("Model saved at ", path)

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
