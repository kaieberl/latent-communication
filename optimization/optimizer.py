# Import relevant libraries
import numpy as np
import cvxpy as cp
from abc import abstractmethod
import sys
sys.path.append('../')
from base_optimizer import Base_Optimizer	

class LinearFitting(Base_Optimizer):
    def __init__(self, z1, z2, lamda):
        """
        Initializes the MatrixFitting class.

        Parameters:
        z1 (np.ndarray): Input data matrix of shape (n_samples, 32)
        z2 (np.ndarray): Output data matrix of shape (n_samples, 32)
        lamda (float): Regularization parameter
        """
        super().__init__(z1, z2, lamda)
        self.A = cp.Variable((self.latent_dim1, self.latent_dim2))
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


class LinearFitting(TransformOptimizer):
    def __init__(self, z1, z2, lamda):
        """
        Initializes the MatrixFitting class.

        Parameters:
        z1 (np.ndarray): Input data matrix of shape (n_samples, 32)
        z2 (np.ndarray): Output data matrix of shape (n_samples, 32)
        lamda (float): Regularization parameter
        """
        super().__init__(z1, z2, lamda)
        self.A = cp.Variable((self.latent_dim1, self.latent_dim2))
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