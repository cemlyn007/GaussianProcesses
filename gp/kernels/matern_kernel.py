import numpy as np

from .kernel import Kernel


class MaternKernel(Kernel):
    def get_covariance_matrix(self, xs: np.ndarray, ys: np.ndarray):
        """
        :param xs: numpy array of size n_1 x l for which each row (x_i) is a data
        point at which the objective function can be evaluated
        :param ys: numpy array of size n_2 x m for which each row (y_j) is a data
        point at which the objective function can be evaluated
        :return: numpy array of size n_1 x n_2 for which the value at position
        (i, j) corresponds to the value of k(x_i, y_j), where k represents the
        kernel used.
        """
        if xs.ndim == 1:
            xs = xs.reshape((len(xs), -1))
        if ys.ndim == 1:
            ys = ys.reshape((len(ys), -1))

        xnorms_2 = np.diag(xs.dot(xs.T)).reshape(len(xs), -1)
        ynorms_2 = np.diag(ys.dot(ys.T)).reshape(len(ys), -1)
        xnorms_2 = xnorms_2 @ np.ones((1, ynorms_2.shape[0]))
        ynorms_2 = np.ones((xnorms_2.shape[0], 1)) @ ynorms_2.T
        tmp_calc = (xnorms_2 + ynorms_2 - 2 * xs @ ys.T)
        tmp_calc = np.sqrt(3 * tmp_calc) / self.length_scale
        return self.amplitude_squared * (1 + tmp_calc) * np.exp(-tmp_calc)
