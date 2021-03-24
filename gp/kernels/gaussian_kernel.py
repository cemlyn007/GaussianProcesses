import numpy as np
from .kernel import Kernel
from scipy.spatial import distance_matrix


class GaussianKernel(Kernel):
    def __init__(self, log_amplitude: float, log_length_scale: float,
                 log_noise_scale: float):
        super(GaussianKernel, self).__init__(log_amplitude, log_length_scale,
                                             log_noise_scale)

    def get_covariance_matrix(self,
                              xs: np.ndarray,
                              ys: np.ndarray,
                              ) -> np.ndarray:
        """
        :param xs: numpy array of size n_1 x l for which each row (x_i) is a data
                    point at which the objective function can be evaluated
        :param ys: numpy array of size n_2 x m for which each row (y_j) is a data
                    point at which the objective function can be evaluated
        :return: numpy array of size n_1 x n_2 for which the value at
                    position (i, j) corresponds to the value of
        k(x_i, y_j), where k represents the kernel used.
        """
        if xs.ndim == 1:
            xs = xs.reshape((len(xs), -1))
        if ys.ndim == 1:
            ys = ys.reshape((len(ys), -1))

        return (
                self.amplitude_squared
                * np.exp(-distance_matrix(xs, ys) ** 2
                         / (2 * self.length_scale ** 2))
        )

    def __call__(self, xs: np.ndarray, ys: np.ndarray) -> np.ndarray:
        return self.get_covariance_matrix(xs, ys)
