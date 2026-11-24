import math

import torch
from torch import nn
from torch.distributions import Categorical

from agents.base import BaseAgent, TsAttnBlock
from agents.critic import InitCritic
from agents.dueling_q import DuelingQDecoder


def _masked_mean(embeddings, mask):
    valid = mask.unsqueeze(-1)
    return embeddings.masked_fill(~valid, 0.0).sum(dim=-2) / valid.sum(dim=-2).clamp_min(1)


def _attention_mask(mask):
    # Give fully padded rows a dummy key. Pooling still uses the original mask.
    first = torch.arange(mask.size(-1), device=mask.device) == 0
    return mask | (~mask.any(dim=-1, keepdim=True) & first)


class _ConstraintEncoder(nn.Module):
    def __init__(self, hidden_dim, dropout, input_dim=2):
        super().__init__()
        self.embedding = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.attention = TsAttnBlock(hidden_dim, n_heads=1, ff_dim=hidden_dim, dropout=dropout)
        self.hidden_dim = hidden_dim

    def forward(self, features, mask):
        batch_size, population_size, n_constraints, _ = features.shape
        if n_constraints == 0 or population_size == 0:
            return features.new_zeros((batch_size, population_size, self.hidden_dim))

        valid = mask.reshape(batch_size * population_size, n_constraints)
        tokens = self.embedding(features.flatten(0, 1))
        tokens = self.attention(tokens, key_padding_mask=~_attention_mask(valid))
        return _masked_mean(tokens, valid).reshape(batch_size, population_size, self.hidden_dim)


class CodeIn(BaseAgent):
    """Neural Hierarchical Infill Sampling with seed grading and infill Q-values."""

    def __init__(self, hidden_dim=128, n_heads=8, ff_dim=256, dropout=0.0):
        super().__init__(hidden_dim, n_heads, ff_dim, dropout)
        self.archive_attention = TsAttnBlock(hidden_dim, n_heads, ff_dim, dropout)
        self.offspring_attention = TsAttnBlock(hidden_dim, n_heads, ff_dim, dropout)
        self.constraint_encoder = _ConstraintEncoder(hidden_dim, dropout, input_dim=3)
        self.archive_fusion = self._fusion_mlp(hidden_dim)
        self.offspring_fusion = self._fusion_mlp(hidden_dim)
        self.init_head = nn.Sequential(
            nn.Linear(2 * hidden_dim + 2, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 6),
        )
        self.critic = InitCritic(hidden_dim)
        self.q_decoder = DuelingQDecoder(hidden_dim, aux_dim=2, dropout=dropout, logit_scale=1.0)

    @staticmethod
    def _fusion_mlp(hidden_dim):
        return nn.Sequential(
            nn.Linear(2 * hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

    @staticmethod
    def _objective_embeddings(features, encoder, objective_attention, dim_mask, population_mask):
        batch_size, n_obj, _, population_size, hidden_dim = features.shape
        if population_size == 0:
            return features.new_zeros((batch_size, 0, hidden_dim))
        encoded = encoder(
            features,
            dim_mask=_attention_mask(dim_mask),
            individual_mask=_attention_mask(population_mask),
        )
        encoded = encoded.masked_fill(~dim_mask.any(dim=1).view(-1, 1, 1, 1), 0.0)
        tokens = encoded.transpose(1, 2).reshape(batch_size * population_size, n_obj, hidden_dim)
        tokens = objective_attention(tokens)
        return tokens.mean(dim=1).reshape(batch_size, population_size, hidden_dim)

    @staticmethod
    def _constraint_vector(value, name, shape, device, dtype):
        vector = torch.as_tensor(value, device=device, dtype=dtype)
        if vector.shape == shape[1:]:
            vector = vector.unsqueeze(0).expand(shape)
        if vector.shape != shape:
            raise ValueError(f"{name} must have shape {shape[1:]} or {shape}, got {tuple(vector.shape)}.")
        return vector

    def _constraint_embedding(self, g, scale, population_mask, reference, std=None, constraint_mask=None):
        batch_size, population_size = population_mask.shape
        if g is None:
            return reference.new_zeros((batch_size, population_size, self.hidden_dim))
        g = self._ensure_batch(torch.as_tensor(g, device=reference.device, dtype=reference.dtype))
        if g.ndim != 3 or g.shape[:2] != population_mask.shape:
            raise ValueError("Constraint values must be [B, N, J] or [N, J] for the corresponding population.")
        if g.size(-1) == 0:
            return reference.new_zeros((batch_size, population_size, self.hidden_dim))
        if scale is None:
            raise ValueError("constraint_scale is required for nonempty constraint inputs.")
        shape = (batch_size, g.size(-1))
        scale = self._constraint_vector(scale, "constraint_scale", shape, g.device, g.dtype)
        valid = torch.ones(shape, dtype=torch.bool, device=g.device)
        if constraint_mask is not None:
            valid = self._constraint_vector(constraint_mask, "constraint_mask", shape, g.device, torch.bool)
        scale = scale.masked_fill(~valid | (scale == 0), 1.0).unsqueeze(1)
        mask = population_mask.unsqueeze(-1) & valid.unsqueeze(1)
        normalized = g.masked_fill(~mask, 0.0) / scale
        violation = normalized.clamp_min(0.0)
        if std is None:
            features = (normalized, violation, (normalized <= 0).to(g.dtype))
        else:
            std = self._ensure_batch(torch.as_tensor(std, device=g.device, dtype=g.dtype))
            if std.shape != g.shape:
                raise ValueError("Constraint means and standard deviations must have the same shape.")
            features = (normalized, std.masked_fill(~mask, 0.0) / scale, violation)
        return self.constraint_encoder(torch.stack(features, dim=-1), mask)

    def encode_archive(self, x_true, y_true, progress, lower_bound, upper_bound, seed_count,
                       g_true=None, constraint_scale=None, archive_mask=None, constraint_mask=None):
        x_true, y_true = self._ensure_batch(x_true), self._ensure_batch(y_true)
        # The archive encoder sees the same information before and after pool generation.
        features, _, dim_mask, mask, _ = self._prepare_inputs(
            x_true, y_true, x_true[:, :0], y_true[:, :0], y_true[:, :0],
            lower_bound, upper_bound, archive_mask=archive_mask, mask_objective_extrema=True,
        )
        objectives = self._objective_embeddings(
            self.W_true(features), self.encoder_true, self.archive_attention, dim_mask, mask
        )
        constraints = self._constraint_embedding(
            g_true, constraint_scale, mask, objectives, constraint_mask=constraint_mask
        )
        h_a = self.archive_fusion(torch.cat((objectives, constraints), dim=-1))
        h_a = h_a.masked_fill(~mask.unsqueeze(-1), 0.0)
        h_A = _masked_mean(h_a, mask)
        p = self._prepare_progress(progress, h_A.device, h_A.dtype, h_A.size(0))
        count = torch.as_tensor(seed_count, device=h_A.device, dtype=h_A.dtype)
        k = count.reshape(-1, 1) / mask.sum(dim=1, keepdim=True).clamp_min(1)
        return h_a, h_A, torch.cat((p, k), dim=-1), mask

    def init_policy(self, **archive_state):
        h_a, h_A, budget, mask = self.encode_archive(**archive_state)
        shared = h_A.unsqueeze(1).expand_as(h_a)
        budget_by_point = budget.unsqueeze(1).expand(-1, h_a.size(1), -1)
        logits = self.init_head(torch.cat((shared, h_a, budget_by_point), dim=-1))
        return logits, self.critic(torch.cat((h_A, budget), dim=-1)), mask

    def choose_seeds(self, *, sample, seed_count, **archive_state):
        logits, value, mask = self.init_policy(seed_count=seed_count, **archive_state)
        if int(seed_count) < 1 or (mask.sum(dim=1) < int(seed_count)).any():
            raise ValueError("seed_count must be between 1 and the number of valid archive points.")
        distribution = Categorical(logits=logits)
        grades = distribution.sample() if sample else logits.argmax(dim=-1)
        log_probs = distribution.log_prob(grades).masked_fill(~mask, 0.0)
        # Shuffle before a stable sort so all ties at the cutoff are sampled uniformly.
        shuffled = torch.rand_like(logits[..., 0]).argsort(dim=1)
        shuffled_scores = (grades + 1).masked_fill(~mask, 0).gather(1, shuffled)
        order = shuffled_scores.argsort(dim=1, descending=True, stable=True)
        indices = shuffled.gather(1, order[:, :int(seed_count)])
        return indices, grades.masked_fill(~mask, -1) + 1, log_probs, value

    def forward(self, x_true, y_true, x_sur, y_sur, sigma_sur, progress, lower_bound, upper_bound,
                seed_count, archive_mask=None, candidate_mask=None, g_true=None, g_sur_mean=None,
                g_sur_std=None, constraint_scale=None, constraint_mask=None):
        h_a, _, budget_features, archive_mask = self.encode_archive(
            x_true, y_true, progress, lower_bound, upper_bound, seed_count,
            g_true=g_true, constraint_scale=constraint_scale, archive_mask=archive_mask,
            constraint_mask=constraint_mask,
        )
        _, features, dim_mask, _, candidate_mask = self._prepare_inputs(
            x_true, y_true, x_sur, y_sur, sigma_sur, lower_bound, upper_bound,
            archive_mask=archive_mask, candidate_mask=candidate_mask, mask_objective_extrema=True,
        )
        objectives = self._objective_embeddings(
            self.W_surr(features), self.encoder_surr, self.offspring_attention, dim_mask, candidate_mask
        )
        if (g_sur_mean is None) != (g_sur_std is None):
            raise ValueError("Supply offspring constraint means and standard deviations together.")
        constraints = self._constraint_embedding(
            g_sur_mean, constraint_scale, candidate_mask, objectives,
            std=g_sur_std, constraint_mask=constraint_mask,
        )
        offspring = self.offspring_fusion(torch.cat((objectives, constraints), dim=-1))
        if h_a.size(1) and offspring.size(1):
            attended = self.cross_space_attn(offspring, h_a, h_a, key_padding_mask=~_attention_mask(archive_mask))
            offspring = torch.where(archive_mask.any(dim=1).view(-1, 1, 1), attended, offspring)
        offspring = offspring.masked_fill(~candidate_mask.unsqueeze(-1), 0.0)
        q = self.q_decoder(
            offspring, budget_features.unsqueeze(1).expand(-1, offspring.size(1), -1),
            budget_features, candidate_mask,
        )
        return {"q_values": q, "candidate_mask": candidate_mask}

    @staticmethod
    def choose_candidate(output, *, sample, temperature=1.0):
        q, mask = output["q_values"], output["candidate_mask"]
        if not mask.any(dim=1).all():
            raise ValueError("Each active infill state needs at least one valid candidate.")
        q = q.masked_fill(~mask, -torch.inf)
        if sample:
            if not math.isfinite(temperature) or temperature <= 0:
                raise ValueError("temperature must be finite and positive.")
            return Categorical(logits=(q - q.amax(dim=1, keepdim=True)) / temperature).sample()
        return q.argmax(dim=1)


ARCHIVE_INPUTS = ("x_true", "y_true", "g_true", "constraint_scale", "progress", "seed_count",
                  "lower_bound", "upper_bound")
OFFSPRING_INPUTS = ("x_sur", "y_sur", "sigma_sur", "g_sur_mean", "g_sur_std")


def tensor_policy_state(state, device, *, include_offspring=False):
    keys = ARCHIVE_INPUTS + OFFSPRING_INPUTS if include_offspring else ARCHIVE_INPUTS
    return {key: torch.as_tensor(state[key], device=device, dtype=torch.float32) for key in keys}


class DiscAF(BaseAgent):
    def __init__(
        self,
        hidden_dim=128,
        n_heads=8,
        ff_dim=256,
        dropout=0.0,
        logit_scale=5.0,
        value_uses_embedding=True,
        epsilon=0.05,
    ):
        super().__init__(
            hidden_dim=hidden_dim,
            n_heads=n_heads,
            ff_dim=ff_dim,
            dropout=dropout,
        )
        self.epsilon = float(epsilon)
        self.q_decoder = DuelingQDecoder(
            hidden_dim=hidden_dim,
            aux_dim=1,
            dropout=dropout,
            logit_scale=logit_scale,
            value_uses_embedding=value_uses_embedding,
        )
        self.true_objective_attn = TsAttnBlock(hidden_dim, n_heads, ff_dim, dropout)
        self.surr_objective_attn = TsAttnBlock(hidden_dim, n_heads, ff_dim, dropout)

    def _aggregate_surrogate_embeddings(self, e_surr, dim_mask):
        e_surr = self._ensure_batch(e_surr)
        if dim_mask is None:
            dim_agg = e_surr.mean(dim=2)
        else:
            dim_mask_f = dim_mask.to(device=e_surr.device, dtype=e_surr.dtype).view(
                e_surr.size(0), 1, e_surr.size(2), 1, 1
            )
            dim_sum = (e_surr * dim_mask_f).sum(dim=2)
            dim_denom = dim_mask_f.sum(dim=2).clamp_min(1.0)
            dim_agg = dim_sum / dim_denom
        return dim_agg.mean(dim=1)

    def _encode_surrogate_without_candidate_attention(self, e_surr, dim_mask):
        """Encode candidate dimensions/objectives without mixing candidates."""
        e_surr = self._ensure_batch(e_surr)
        bsz, n_obj, n_dim, n_surr, hidden_dim = e_surr.shape

        x = e_surr.permute(0, 1, 3, 2, 4).reshape(bsz * n_obj * n_surr, n_dim, hidden_dim)
        x = self.encoder_surr.position(x)
        dim_padding_mask = None
        if dim_mask is not None:
            dim_mask = dim_mask.to(device=x.device, dtype=torch.bool)
            dim_padding_mask = (~dim_mask).unsqueeze(1).unsqueeze(1)
            dim_padding_mask = dim_padding_mask.expand(-1, n_obj, n_surr, -1)
            dim_padding_mask = dim_padding_mask.reshape(bsz * n_obj * n_surr, n_dim)
        x = self.encoder_surr.cross_dimension(x, key_padding_mask=dim_padding_mask)
        x = x.reshape(bsz, n_obj, n_surr, n_dim, hidden_dim)

        if dim_mask is None:
            return x.mean(dim=3)

        dim_mask_expanded = dim_mask.to(device=x.device, dtype=x.dtype).view(bsz, 1, 1, n_dim, 1)
        masked_sum = (x * dim_mask_expanded).sum(dim=3)
        denom = dim_mask_expanded.sum(dim=3).clamp_min(1.0)
        return masked_sum / denom

    def _prepare_progress_for_candidates(self, h_surr, progress):
        h_surr = self._ensure_batch(h_surr)

        progress_state = self._prepare_progress(
            progress=progress,
            device=h_surr.device,
            dtype=h_surr.dtype,
            batch_size=h_surr.size(0),
        )

        return progress_state.unsqueeze(1).expand(-1, h_surr.size(1), -1)

    def _actor_logits(self, h_surr, progress, candidate_mask=None):
        h_surr = self._ensure_batch(h_surr)

        progress_state = self._prepare_progress(
            progress=progress,
            device=h_surr.device,
            dtype=h_surr.dtype,
            batch_size=h_surr.size(0),
        )

        progress_cand = progress_state.unsqueeze(1).expand(-1, h_surr.size(1), -1)

        q_values = self.q_decoder(
            h_cand=h_surr,
            aux_cand=progress_cand,
            aux_state=progress_state,
            candidate_mask=candidate_mask,
        )

        return q_values

    def _epsilon_greedy_ranking(self, q_values, epsilon=0.05, candidate_mask=None):
        bsz, n_cand = q_values.shape
        device = q_values.device

        if candidate_mask is not None:
            q_values = q_values.masked_fill(~candidate_mask, -1e9)

        greedy_ranking = torch.argsort(q_values, dim=-1, descending=True)

        random_rankings = []
        for b in range(bsz):
            if candidate_mask is None:
                perm = torch.randperm(n_cand, device=device)
            else:
                valid_idx = torch.nonzero(candidate_mask[b], as_tuple=False).squeeze(-1)
                invalid_idx = torch.nonzero(~candidate_mask[b], as_tuple=False).squeeze(-1)
                valid_perm = valid_idx[torch.randperm(valid_idx.numel(), device=device)]
                perm = torch.cat([valid_perm, invalid_idx], dim=0)
            random_rankings.append(perm)

        random_ranking = torch.stack(random_rankings, dim=0)
        use_random = torch.rand(bsz, device=device) < float(epsilon)
        ranking = torch.where(use_random.unsqueeze(1), random_ranking, greedy_ranking)
        return ranking

    def decode_ranking(
        self,
        h_surr,
        progress,
        target_ranking=None,
        decode_type="epsilon_greedy",
        max_decode_steps=None,
        candidate_mask=None,
        epsilon=None,
    ):
        h_surr = self._ensure_batch(h_surr)

        q_values = self._actor_logits(
            h_surr=h_surr,
            progress=progress,
            candidate_mask=candidate_mask,
        )

        if epsilon is None:
            epsilon = self.epsilon

        if target_ranking is not None:
            ranking = target_ranking
        elif decode_type == "epsilon_greedy":
            ranking = self._epsilon_greedy_ranking(
                q_values=q_values,
                epsilon=epsilon,
                candidate_mask=candidate_mask,
            )
        elif decode_type in ["q_greedy", "greedy"]:
            ranking = torch.argsort(q_values, dim=-1, descending=True)
        elif decode_type == "softmax_sample":
            decode_steps = q_values.size(1) if max_decode_steps is None else min(
                int(max_decode_steps), q_values.size(1)
            )
            ranking = self._sample_actions_without_replacement(q_values, decode_steps)
        else:
            raise ValueError(f"Unknown decode_type: {decode_type}")

        if max_decode_steps is not None:
            ranking = ranking[:, : min(int(max_decode_steps), ranking.size(1))]

        return {
            "logits": q_values,
            "q_values": q_values,
            "ranking": ranking,
        }

    def _sample_actions_without_replacement(self, logits, k):
        bsz, n_sur = logits.shape
        selected_mask = torch.zeros(bsz, n_sur, dtype=torch.bool, device=logits.device)
        actions = []
        decode_steps = min(int(k), n_sur)

        for _ in range(decode_steps):
            masked_logits = logits.masked_fill(selected_mask, -1e9)
            dist = torch.distributions.Categorical(logits=masked_logits)
            chosen_idx = dist.sample()
            actions.append(chosen_idx.unsqueeze(1))
            selected_mask = selected_mask.scatter(1, chosen_idx.unsqueeze(1), True)

        return torch.cat(actions, dim=1)

    def encode(
        self,
        x_true,
        y_true,
        x_sur,
        y_sur,
        sigma_sur,
        progress,
        lower_bound,
        upper_bound,
        archive_mask=None,
        candidate_mask=None,
    ):
        m_true, m_surr, dim_mask, archive_mask, candidate_mask = self._prepare_inputs(
            x_true=x_true,
            y_true=y_true,
            x_sur=x_sur,
            y_sur=y_sur,
            sigma_sur=sigma_sur,
            lower_bound=lower_bound,
            upper_bound=upper_bound,
            archive_mask=archive_mask,
            candidate_mask=candidate_mask,
        )

        e_true = self.W_true(m_true)
        e_surr = self.W_surr(m_surr)

        s_true = self.encoder_true(e_true, dim_mask=dim_mask, individual_mask=archive_mask)
        bsz, n_obj, n_true, hidden_dim = s_true.shape
        s_true_obj = s_true.permute(0, 2, 1, 3).reshape(bsz * n_true, n_obj, hidden_dim)
        s_true_obj = self.true_objective_attn(s_true_obj)
        s_true = s_true_obj.reshape(bsz, n_true, n_obj, hidden_dim).permute(0, 2, 1, 3)
        h_true = s_true.mean(dim=1)

        s_surr = self._encode_surrogate_without_candidate_attention(e_surr, dim_mask)
        n_surr = int(s_surr.shape[2])
        s_surr_obj = s_surr.permute(0, 2, 1, 3).reshape(bsz * n_surr, n_obj, hidden_dim)
        s_surr_obj = self.surr_objective_attn(s_surr_obj)
        s_surr = s_surr_obj.reshape(bsz, n_surr, n_obj, hidden_dim).permute(0, 2, 1, 3)
        h_surr_raw = s_surr.mean(dim=1)
        h_surr = self.cross_space_attn(
            h_surr_raw,
            h_true,
            h_true,
            key_padding_mask=(~archive_mask),
        )

        progress = self._prepare_progress(
            progress=progress,
            device=h_true.device,
            dtype=h_true.dtype,
            batch_size=h_true.size(0),
        )

        return {
            "H_true": h_true,
            "H_surr": h_surr,
            "progress": progress,
        }

    def forward(
        self,
        x_true,
        y_true,
        x_sur,
        y_sur,
        sigma_sur,
        progress,
        lower_bound,
        upper_bound,
        archive_mask=None,
        candidate_mask=None,
        target_ranking=None,
        decode_type="epsilon_greedy",
        max_decode_steps=None,
        epsilon=None,
    ):
        encoded = self.encode(
            x_true=x_true,
            y_true=y_true,
            x_sur=x_sur,
            y_sur=y_sur,
            sigma_sur=sigma_sur,
            progress=progress,
            lower_bound=lower_bound,
            upper_bound=upper_bound,
            archive_mask=archive_mask,
            candidate_mask=candidate_mask,
        )

        decoded = self.decode_ranking(
            h_surr=encoded["H_surr"],
            progress=encoded["progress"],
            target_ranking=target_ranking,
            decode_type=decode_type,
            max_decode_steps=max_decode_steps,
            candidate_mask=candidate_mask,
            epsilon=epsilon,
        )

        encoded.update(decoded)
        return encoded
