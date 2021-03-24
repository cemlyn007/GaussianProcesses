from gp.kernels.kernel import Kernel
import pytest


class TestKernel:

    def test_instantiate_kernel(self, log_amplitude: float,
                                log_length_scale: float, log_noise_scale: float):
        with pytest.raises(TypeError) as e:
            Kernel(log_amplitude, log_length_scale, log_noise_scale)
