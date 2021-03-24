import numpy as np
import pytest
from gp.kernels.kernel import Kernel
from gp.kernels.gaussian_kernel import GaussianKernel
from gp.kernels.matern_kernel import MaternKernel
from gp.gaussian_process import GaussianProcess


@pytest.fixture
def log_amplitude() -> float:
    return -1.


@pytest.fixture
def log_length_scale() -> float:
    return 0.


@pytest.fixture
def log_noise_scale() -> float:
    return -1.


@pytest.fixture
def log_parameters(log_amplitude, log_length_scale, log_noise_scale) -> (float, float, float):
    return log_amplitude, log_length_scale, log_noise_scale


@pytest.fixture
def gaussian_kernel(log_amplitude, log_length_scale, log_noise_scale) -> GaussianKernel:
    return GaussianKernel(log_amplitude, log_length_scale, log_noise_scale)


@pytest.fixture
def matern_kernel(log_amplitude, log_length_scale, log_noise_scale) -> MaternKernel:
    return MaternKernel(log_amplitude, log_length_scale, log_noise_scale)


@pytest.fixture
def random_data_points():
    return np.random.randn(10, 6)


@pytest.fixture
def random_objective_function_values():
    return np.random.randn(10, 1)


@pytest.fixture
def gaussian_process(gaussian_kernel: GaussianKernel):
    return GaussianProcess(gaussian_kernel)
