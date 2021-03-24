import numpy as np

from gp.kernels.gaussian_kernel import GaussianKernel


class TestGaussianKernel:
    def test_set_parameters(self, gaussian_kernel: GaussianKernel,
                            log_amplitude: float, log_length_scale: float,
                            log_noise_scale: float):
        gaussian_kernel.log_amplitude = log_amplitude
        gaussian_kernel.log_length_scale = log_length_scale
        gaussian_kernel.log_noise_scale = log_noise_scale
        assert gaussian_kernel.log_amplitude == log_amplitude
        assert gaussian_kernel.log_length_scale == log_length_scale
        assert gaussian_kernel.log_noise_scale == log_noise_scale

    def test_log_amplitude(self, gaussian_kernel: GaussianKernel):
        assert isinstance(gaussian_kernel.log_amplitude, float)
        gaussian_kernel.log_amplitude = np.log(2)
        assert gaussian_kernel.log_amplitude == np.log(2)

    def test_log_length_scale(self, gaussian_kernel: GaussianKernel):
        assert isinstance(gaussian_kernel.log_length_scale, float)
        gaussian_kernel.log_length_scale = np.log(2)
        assert gaussian_kernel.log_length_scale == np.log(2)

    def test_log_noise_scale(self, gaussian_kernel: GaussianKernel):
        assert isinstance(gaussian_kernel.log_noise_scale, float)
        gaussian_kernel.log_noise_scale = np.log(2)
        assert gaussian_kernel.log_noise_scale == np.log(2)

    def test_amplitude_squared(self, gaussian_kernel: GaussianKernel):
        assert isinstance(gaussian_kernel.amplitude_squared, float)
        assert gaussian_kernel.amplitude_squared == np.exp(gaussian_kernel.log_amplitude * 2)

    def test_length_scale(self, gaussian_kernel: GaussianKernel):
        assert isinstance(gaussian_kernel.length_scale, float)
        assert gaussian_kernel.length_scale == np.exp(gaussian_kernel.log_length_scale)

    def test_noise_scale_squared(self, gaussian_kernel: GaussianKernel):
        assert isinstance(gaussian_kernel.noise_scale_squared, float)
        assert gaussian_kernel.noise_scale_squared == np.exp(gaussian_kernel.log_noise_scale * 2)

    def test_log_parameters(self, gaussian_kernel: GaussianKernel):
        log_amplitude, log_length_scale, log_noise_scale = gaussian_kernel.log_parameters
        assert gaussian_kernel.log_amplitude == log_amplitude
        assert gaussian_kernel.log_length_scale == log_length_scale
        assert gaussian_kernel.log_noise_scale == log_noise_scale

    def test_get_covariance_matrix(self, gaussian_kernel: GaussianKernel):
        cov = gaussian_kernel(np.ones((10, 2)), np.ones((10, 2)))
        assert cov.shape == (10, 10)
        assert (cov.diagonal() == (gaussian_kernel.amplitude_squared / (gaussian_kernel.length_scale ** 2))).all()
