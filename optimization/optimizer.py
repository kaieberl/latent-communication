import numpy as np
import cvxpy as cp
from base_optimizer import Base_Optimizer


class AffineFitting(Base_Optimizer):
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


class LinearFitting(Base_Optimizer):
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
    
