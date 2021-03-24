import pytest
from gp.gaussian_process import GaussianProcess
from gp.kernels.gaussian_kernel import GaussianKernel


class TestObjectiveFunction:

    def instantiate_objective_function(self):
        kernel = GaussianKernel(-1., 0., -1.)
        with pytest.raises(TypeError):
            GaussianProcess(kernel)

