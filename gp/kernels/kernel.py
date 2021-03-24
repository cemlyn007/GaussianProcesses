import abc

import numpy as np


class Kernel(metaclass=abc.ABCMeta):
    def __init__(self, log_amplitude: float, log_length_scale: float,
                 log_noise_scale: float):
        self._log_amplitude = log_amplitude
        self._log_length_scale = log_length_scale
        self._log_noise_scale = log_noise_scale

    def set_parameters(self, log_amplitude: float, log_length_scale: float,
                       log_noise_scale: float) -> None:
        self._log_amplitude = log_amplitude
        self._log_length_scale = log_length_scale
        self._log_noise_scale = log_noise_scale

    @property
    def log_amplitude(self) -> float:
        return self._log_amplitude

    @property
    def log_length_scale(self) -> float:
        return self._log_length_scale

    @property
    def log_noise_scale(self) -> float:
        return self._log_noise_scale

    @log_amplitude.setter
    def log_amplitude(self, log_amplitude: float):
        log_amplitude = np.clip(log_amplitude, -3, 3)
        self._log_amplitude = log_amplitude

    @log_length_scale.setter
    def log_length_scale(self, log_length_scale: float):
        log_length_scale = np.clip(log_length_scale, -3, 3)
        self._log_length_scale = log_length_scale

    @log_noise_scale.setter
    def log_noise_scale(self, log_noise_scale: float):
        log_noise_scale = np.clip(log_noise_scale, -3, 3)
        self._log_noise_scale = log_noise_scale

    @property
    def amplitude_squared(self) -> float:
        return np.exp(self.log_amplitude * 2)

    @property
    def length_scale(self) -> float:
        return np.exp(self.log_length_scale)

    @property
    def noise_scale_squared(self) -> float:
        return np.exp(self.log_noise_scale * 2)

    @property
    def log_parameters(self) -> (float, float, float):
        return self.log_amplitude, self.log_length_scale, self.log_noise_scale

    @abc.abstractmethod
    def get_covariance_matrix(self, xs: np.ndarray, ys: np.ndarray
                              ) -> np.ndarray:
        """
        :param xs: numpy array of size n_1 x l for which each row (x_i) is a data
        point at which the objective function can be evaluated
        :param ys: numpy array of size n_2 x m for which each row (y_j) is a data
         point at which the objective function can be evaluated
        :return: numpy array of size n_1 x n_2 for which the value at position
        (i, j) corresponds to the value of k(x_i, y_j), where k represents the
        kernel used.
        """
        pass

    def __call__(self, xs: np.ndarray, ys: np.ndarray):
        return self.get_covariance_matrix(xs, ys)
