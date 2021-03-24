import numpy as np
from gp.kernels.matern_kernel import MaternKernel


class TestMaternKernel:
    def test_get_covariance_matrix(self, matern_kernel: MaternKernel):
        cov = matern_kernel(np.ones((10, 2)), np.ones((10, 2)))
        assert cov.shape == (10, 10)
        assert (cov.diagonal() == (matern_kernel.amplitude_squared / (matern_kernel.length_scale ** 2))).all()
