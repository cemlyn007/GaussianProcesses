from gp.gaussian_process import GaussianProcess
import numpy as np


class TestGaussianProcess:
    def test_array_dataset(self, gaussian_process: GaussianProcess,
                           random_data_points: np.ndarray,
                           random_objective_function_values: np.ndarray):
        assert gaussian_process.array_dataset.size == 0
        gaussian_process.initialise_dataset(random_data_points, random_objective_function_values)
        assert gaussian_process.array_dataset.shape == random_data_points.shape

    def test_array_objective_function_values(self, gaussian_process: GaussianProcess,
                                             random_data_points: np.ndarray,
                                             random_objective_function_values: np.ndarray):
        assert gaussian_process.array_objective_function_values.size == 0
        gaussian_process.initialise_dataset(random_data_points, random_objective_function_values)
        assert gaussian_process.array_objective_function_values.shape == random_objective_function_values.shape

    def test_set_kernel_parameters(self, gaussian_process: GaussianProcess,
                                   log_amplitude, log_length_scale,
                                   log_noise_scale):
        gaussian_process.set_kernel_parameters(log_amplitude, log_length_scale, log_noise_scale)
        assert gaussian_process._kernel.log_parameters == (log_amplitude, log_length_scale, log_noise_scale)

    def test_update_covariance_matrix(self, gaussian_process: GaussianProcess,
                                      random_data_points: np.ndarray,
                                      random_objective_function_values: np.ndarray):
        gaussian_process.initialise_dataset(random_data_points, random_objective_function_values)
        assert gaussian_process._covariance_matrix.size > 0

    def test_add_data_point(self, gaussian_process: GaussianProcess,
                            random_data_points: np.ndarray,
                            random_objective_function_values: np.ndarray):
        gaussian_process.initialise_dataset(random_data_points, random_objective_function_values)
        new_data_point = np.random.randn(*random_data_points[:1, :].shape)
        new_objective_function_value = np.random.randn(*random_objective_function_values[:1, :].shape)
        gaussian_process.add_data_point(new_data_point, new_objective_function_value)
        assert len(gaussian_process.array_dataset) == len(random_data_points) + 1

    def test_initialise_dataset(self, gaussian_process: GaussianProcess,
                                random_data_points: np.ndarray,
                                random_objective_function_values: np.ndarray):
        assert gaussian_process.array_dataset.size == 0
        gaussian_process.initialise_dataset(random_data_points, random_objective_function_values)
        assert gaussian_process.array_dataset.size > 0

    def test_optimise_parameters(self, gaussian_process: GaussianProcess,
                                 random_data_points: np.ndarray,
                                 random_objective_function_values: np.ndarray):
        gaussian_process.initialise_dataset(random_data_points, random_objective_function_values)
        original_params = gaussian_process._kernel.log_parameters
        gaussian_process.optimise_parameters(disp=False)
        assert original_params != gaussian_process._kernel.log_parameters

    def test_get_negative_log_marginal_likelihood(self, gaussian_process: GaussianProcess,
                                                  random_data_points: np.ndarray,
                                                  random_objective_function_values: np.ndarray):
        gaussian_process.initialise_dataset(random_data_points, random_objective_function_values)
        params = gaussian_process._kernel.log_parameters
        value = gaussian_process.get_negative_log_marginal_likelihood(*params)
        assert isinstance(value, float)

    def test_get_gradient_negative_log_marginal_likelihood(self, gaussian_process: GaussianProcess,
                                                           random_data_points: np.ndarray,
                                                           random_objective_function_values: np.ndarray):
        gaussian_process.initialise_dataset(random_data_points, random_objective_function_values)
        params = gaussian_process._kernel.log_parameters
        gradient = gaussian_process.get_gradient_negative_log_marginal_likelihood(*params)
        assert isinstance(gradient, np.ndarray)
        assert gradient.shape == (3,)

    def test_mean(self, gaussian_process: GaussianProcess,
                  random_data_points: np.ndarray,
                  random_objective_function_values: np.ndarray):
        gaussian_process.initialise_dataset(random_data_points, random_objective_function_values)
        mean = gaussian_process.mean(gaussian_process.array_dataset)
        assert isinstance(mean, np.ndarray)
        assert mean.shape == (len(gaussian_process.array_dataset), 1)

    def test_std(self, gaussian_process: GaussianProcess,
                 random_data_points: np.ndarray,
                 random_objective_function_values: np.ndarray):
        gaussian_process.initialise_dataset(random_data_points, random_objective_function_values)
        std = gaussian_process.std(gaussian_process.array_dataset)
        assert isinstance(std, np.ndarray)
        assert std.shape == (len(gaussian_process.array_dataset), 1)

    def test_get_sample(self, gaussian_process: GaussianProcess,
                        random_data_points: np.ndarray,
                        random_objective_function_values: np.ndarray):
        gaussian_process.initialise_dataset(random_data_points, random_objective_function_values)
        new_data_points = np.random.randn(*random_data_points.shape)
        assert gaussian_process.get_sample(new_data_points).shape == new_data_points.shape[:1]

    def test_get_gp_mean_std(self, gaussian_process: GaussianProcess,
                             random_data_points: np.ndarray,
                             random_objective_function_values: np.ndarray):
        gaussian_process.initialise_dataset(random_data_points, random_objective_function_values)
        mean, std = gaussian_process.get_gp_mean_std(gaussian_process.array_dataset)
        assert isinstance(mean, np.ndarray)
        assert isinstance(std, np.ndarray)
        assert mean.shape == gaussian_process.array_objective_function_values.shape
        assert std.shape == gaussian_process.array_objective_function_values.shape

    def test_get_mse(self, gaussian_process: GaussianProcess,
                     random_data_points: np.ndarray,
                     random_objective_function_values: np.ndarray):
        gaussian_process.initialise_dataset(random_data_points, random_objective_function_values)
        mse = gaussian_process.get_mse(gaussian_process.array_dataset,
                                       gaussian_process.array_objective_function_values)
        assert isinstance(mse, float)
        assert mse > 0

    def test_get_log_predictive_density(self, gaussian_process: GaussianProcess,
                                        random_data_points: np.ndarray,
                                        random_objective_function_values: np.ndarray):
        gaussian_process.initialise_dataset(random_data_points, random_objective_function_values)
        log_predictive_density = gaussian_process.get_log_predictive_density(gaussian_process.array_dataset,
                                                                             gaussian_process.array_objective_function_values)
        assert isinstance(log_predictive_density, float)
