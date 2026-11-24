import copy
import inspect
import unittest

import torch

from agents.codein import CodeIn
from train import TrainConfig, _transition_from_states, compute_ddqn_loss


class CodeInFusionTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(17)
        self.agent = CodeIn(hidden_dim=16, n_heads=2, ff_dim=32).eval()
        self.inputs = {
            "x_true": torch.rand(2, 5, 3),
            "y_true": torch.rand(2, 5, 2),
            "x_sur": torch.rand(2, 3, 3),
            "y_sur": torch.rand(2, 3, 2),
            "sigma_sur": torch.rand(2, 3, 2),
            "g_true": torch.randn(2, 5, 2),
            "g_sur_mean": torch.randn(2, 3, 2),
            "g_sur_std": torch.rand(2, 3, 2),
            "constraint_scale": torch.tensor([[2., 3.], [4., 5.]]),
            "progress": torch.tensor([[.3], [.5]]),
            "seed_count": torch.tensor([2, 2]),
            "lower_bound": torch.zeros(2, 3),
            "upper_bound": torch.ones(2, 3),
            "archive_mask": torch.tensor([[True, True, True, False, False], [True] * 5]),
            "candidate_mask": torch.tensor([[True, True, False], [True] * 3]),
        }

    def archive_inputs(self, inputs):
        keys = ("x_true", "y_true", "g_true", "constraint_scale", "progress",
                "seed_count", "lower_bound", "upper_bound", "archive_mask")
        return {key: inputs[key] for key in keys}

    def test_padding_pooling_and_selection(self):
        with torch.no_grad():
            output = self.agent(**self.inputs)
            self.assertTrue(torch.isfinite(output["q_values"]).all())
            embeddings, pooled, _, mask = self.agent.encode_archive(**self.archive_inputs(self.inputs))
            torch.testing.assert_close(pooled[0], embeddings[0, :3].mean(dim=0))
            torch.testing.assert_close(pooled[1], embeddings[1].mean(dim=0))

            # Changing masked values must have no effect on valid outputs or pooling.
            changed = {key: value.clone() for key, value in self.inputs.items()}
            for key in ("x_true", "y_true", "g_true"):
                changed[key][~mask] = 10000.
            for key in ("x_sur", "y_sur", "sigma_sur", "g_sur_mean", "g_sur_std"):
                changed[key][~output["candidate_mask"]] = -10000.
            torch.testing.assert_close(self.agent(**changed)["q_values"], output["q_values"])
            torch.testing.assert_close(
                self.agent.encode_archive(**self.archive_inputs(changed))[1], pooled,
            )

            # Removing padding must also preserve the result for this example.
            trimmed = {key: value[:1] for key, value in self.inputs.items()}
            for key in ("x_true", "y_true", "g_true", "archive_mask"):
                trimmed[key] = trimmed[key][:, :3]
            for key in ("x_sur", "y_sur", "sigma_sur", "g_sur_mean", "g_sur_std", "candidate_mask"):
                trimmed[key] = trimmed[key][:, :2]
            torch.testing.assert_close(self.agent(**trimmed)["q_values"], output["q_values"][:1, :2])

            for sample in (False, True):
                for _ in range(20):
                    selected = self.agent.choose_candidate(output, sample=sample)
                    self.assertTrue(output["candidate_mask"].gather(1, selected[:, None]).all())

    def test_permutations(self):
        with torch.no_grad():
            expected = self.agent(**self.inputs)["q_values"]
            permuted = dict(self.inputs)
            order = torch.tensor([3, 0, 4, 1, 2])
            for key in ("x_true", "y_true", "g_true", "archive_mask"):
                permuted[key] = self.inputs[key][:, order]
            torch.testing.assert_close(self.agent(**permuted)["q_values"], expected)
            order = torch.tensor([2, 0, 1])
            permuted = dict(self.inputs)
            for key in ("x_sur", "y_sur", "sigma_sur", "g_sur_mean", "g_sur_std", "candidate_mask"):
                permuted[key] = self.inputs[key][:, order]
            torch.testing.assert_close(self.agent(**permuted)["q_values"], expected[:, order])

    def test_ddqn_gradients_and_no_cross_attention(self):
        self.assertFalse(any("cross_space_attn" in name for name, _ in self.agent.named_modules()))
        self.assertNotIn("cross_space_attn", inspect.getsource(CodeIn.forward))
        transitions = []
        for index in range(2):
            state = {}
            for key, value in self.inputs.items():
                if key not in ("archive_mask", "candidate_mask"):
                    state[key] = value[index].numpy()
            for key in ("x_true", "y_true", "g_true"):
                state[key] = state[key][self.inputs["archive_mask"][index].numpy()]
            for key in ("x_sur", "y_sur", "sigma_sur", "g_sur_mean", "g_sur_std"):
                state[key] = state[key][self.inputs["candidate_mask"][index].numpy()]
            state["y_true_history"] = state["y_true"][:, None, :]
            state["sigma_true_history"] = state["y_true_history"] * 0
            transitions.append(_transition_from_states(state, 0, 2., state, False))
        batch = tuple(zip(*transitions))
        target = copy.deepcopy(self.agent)
        loss, _ = compute_ddqn_loss(self.agent, target, batch, TrainConfig(device="cpu"))
        self.assertTrue(torch.isfinite(loss))
        loss.backward()
        for name, parameter in self.agent.named_parameters():
            if parameter.grad is not None:
                self.assertTrue(torch.isfinite(parameter.grad).all(), name)
        for module in (self.agent.infill_fusion, self.agent.encoder_true,
                       self.agent.archive_attention, self.agent.archive_fusion,
                       self.agent.constraint_encoder):
            gradient = sum(p.grad.abs().sum().item() for p in module.parameters() if p.grad is not None)
            self.assertGreater(gradient, 0., type(module).__name__)
        self.assertTrue(all(p.grad is None for p in target.parameters()))


if __name__ == "__main__":
    unittest.main()
