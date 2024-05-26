# Import relevant libraries
import numpy as np
import cvxpy as cp
from abc import abstractmethod


class TransformOptimizer(self,z1,z2):
    def __init__(self, z1, z2, lamda):
        """
        Initializes the TransformOptimizer class.

        Parameters:
        z1 (np.ndarray): Input data matrix of shape (n_samples, latent_dim1)
        z2 (np.ndarray): Output data matrix of shape (n_samples, latent_dim2)
        lamda (float): Regularization parameter
        """
        self.z1 = z1
        self.z2 = z2
        self.lamda = lamda
        self.latent_dim1 = z1.shape[1]
        self.latent_dim2 = z2.shape[1]

    @abstractmethod
    def define_loss(self):
        """
        Defines the loss function for the optimization problem.
        """
        pass

    def define_problem(self):
        """
        Defines the optimization problem.
        """
        loss = self.define_loss()
        objective = cp.Minimize(loss)
        self.problem = cp.Problem(objective)

    def solve_problem(self):
        """
        Solves the optimization problem.
        """
        if self.problem is None:
            self.define_problem()
        self.problem.solve()

    @abstractmethod
    def get_results(self):
        """
        Returns the results of the optimization problem.

        Returns:
        float: Optimal value of the problem
        np.ndarray: Optimal values for A
        np.ndarray: Optimal values for b
        """
        pass


    @abstractmethod    
    def save_results(self, path):
        """
        Saves the results of the optimization problem.

        Parameters:
        path (str): Path to save the results
        """
        pass

    @abstractmethod    
    def print_results(self):
        """
        Prints the results of the optimization problem.
        """
        pass



class AffineFitting(TransformOptimizer):
    def __init__(self, z1, z2, lamda):
        """
        Initializes the AffineFitting class.

        Parameters:
        z1 (np.ndarray): Input data matrix of shape (n_samples, latent_dim1)
        z2 (np.ndarray): Output data matrix of shape (n_samples, latent_dim2)
        lamda (float): Regularization parameter
        """
        super().__init__(z1, z2, lamda)
        self.A_aff = cp.Variable((self.latent_dim1, self.latent_dim2))
        self.b_aff = cp.Variable(self.latent_dim2)
        self.problem_aff = None

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
        if self.problem_aff is None:
            raise Exception("The problem has not been solved yet.")
        return self.problem_aff.value, self.A_aff.value, self.b_aff.value

    def print_results(self):
        """
        Prints the results of the optimization problem.
        """
        opt_value, A_aff_opt, b_aff_opt = self.get_results()
        print("Optimal value: ", opt_value)
        print(A_aff_opt)
        print(b_aff_opt)




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
