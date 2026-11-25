import contextlib
import io
import os
import tempfile
import unittest
from concurrent.futures import ProcessPoolExecutor
from unittest.mock import patch

import numpy as np
from scipy.linalg import solve_triangular

import train
from surrogate.surrogate import GPSurrogateModel


class RolloutOptimizationTests(unittest.TestCase):
    def test_gp_uncertainty_matches_previous_formula(self):
        rng = np.random.default_rng(31)
        x = rng.random((20, 3))
        y = np.column_stack((np.sin(x.sum(axis=1)), (x * x).sum(axis=1)))
        model = GPSurrogateModel(n_var=3, n_obj=2)
        for gp in model.gps:
            gp.optimizer = None
        model.fit(x, y)
        query = np.vstack((x[:3], rng.random((7, 3))))
        expected = []
        for gp in model.gps:
            k = gp.kernel_(query, gp.X_train_)
            inverse_l = solve_triangular(gp.L_.T, np.eye(len(x)))
            inverse_k = inverse_l @ inverse_l.T
            variance = gp.kernel_.diag(query) - np.einsum("ij,ij->i", k @ inverse_k, k)
            expected.append(np.sqrt(np.maximum(variance, 0)))
        expected = np.stack(expected, axis=1).astype(np.float32)
        # Near training points the variance is close to zero, so use absolute tolerance.
        np.testing.assert_allclose(model.predict_std(query), expected + 1e-6, atol=2e-7, rtol=1e-5)
        out = model.evaluate(query, std=True, calc_gradient=True)
        np.testing.assert_allclose(out["S"], expected, atol=2e-7, rtol=1e-5)
        for value in out.values():
            if value is not None:
                self.assertTrue(np.isfinite(value).all())
        np.testing.assert_allclose(model.predict_std(query), out["S"] + 1e-6)
        np.testing.assert_array_equal(model.predict_std(query[:0]).shape, (0, 2))

    def test_reset_fits_each_surrogate_once(self):
        cfg = train.TrainConfig(init_size=6, max_fe=8, offspring_size=6, saea_steps=2).__dict__
        env = train.DiscSAEAEnv("DAS-CMOP1", 3, 0, cfg)
        with patch.object(train, "fit_gp_surrogates", wraps=train.fit_gp_surrogates) as fit:
            state = env.reset()
        # One objective surrogate and one constraint surrogate.
        self.assertEqual(fit.call_count, 2)
        self.assertEqual(state["x_sur"].shape, (6, 3))
        self.assertTrue(np.isfinite(state["sigma_sur"]).all())

    def test_training_reuses_process_pool_across_epochs(self):
        previous_directory = os.getcwd()
        output = io.StringIO()
        with tempfile.TemporaryDirectory() as directory:
            try:
                os.chdir(directory)
                with patch.object(train, "build_training_env_specs", return_value=[
                    {"problem_name": "MW1", "dim": 3},
                ]), patch.object(train, "ProcessPoolExecutor", wraps=ProcessPoolExecutor) as pool:
                    with contextlib.redirect_stdout(output):
                        train.train_codein(
                            problem_name="DAS", epoch=2, num_workers=1, init_fe=6, max_fe=8,
                            updates_per_epoch=1, hidden_dim=16, device="cpu", num_seeds=2,
                            cuda_cleanup_before_update=False, cuda_cleanup_after_update=False,
                        )
                    self.assertEqual(pool.call_count, 1)
            finally:
                os.chdir(previous_directory)
        lines = [line for line in output.getvalue().splitlines() if "interact parallel" in line]
        self.assertEqual(len(lines), 2)
        pids = [line.split("pid_list=")[1].split(" | ")[0] for line in lines]
        self.assertEqual(pids[0], pids[1])


if __name__ == "__main__":
    unittest.main()
