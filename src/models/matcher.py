"""GRALE-style soft and hard matching with explicit axis conventions."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from scipy.optimize import linear_sum_assignment
from torch import Tensor, nn

from any2graph_v2.losses.feature_diffusion import pairwise_squared_l2
from any2graph_v2.parameter import MatcherParameters, ObjectiveParameters


@dataclass
class MatchResult:
    """Differentiable transport plan with rows predicted and columns target."""

    plan: Tensor
    cost: Tensor
    diagnostics: dict[str, Tensor]


@dataclass
class HardMatchResult:
    """Discrete target-to-predicted indices and equivalent `T[p,t]` matrices."""

    permutations: Tensor
    matrices: Tensor
    cost: Tensor


def permutations_to_matrices(permutations: Tensor) -> Tensor:
    """Convert `perm[b,t] = p` into one-hot matrices `T[b,p,t]`."""

    if permutations.ndim != 2:
        raise ValueError("permutations must have shape [batch, target].")
    size = permutations.shape[1]
    if torch.any(permutations < 0) or torch.any(permutations >= size):
        raise ValueError("permutations contain an out-of-range index.")
    sorted_values = permutations.sort(dim=1).values
    expected = torch.arange(size, device=permutations.device).expand_as(permutations)
    if not torch.equal(sorted_values, expected):
        raise ValueError("Every row must contain each predicted index exactly once.")
    return torch.nn.functional.one_hot(permutations, num_classes=size).transpose(1, 2)


def marginal_diagnostics(plan: Tensor) -> dict[str, Tensor]:
    """Summarize deviation from unit row and column transport marginals."""
    row_error = (plan.sum(dim=2) - 1.0).abs()
    column_error = (plan.sum(dim=1) - 1.0).abs()
    return {
        "row_marginal_max_error": row_error.amax(),
        "row_marginal_mean_error": row_error.mean(),
        "column_marginal_max_error": column_error.amax(),
        "column_marginal_mean_error": column_error.mean(),
    }

def sinkhorn_unrolling(cost: Tensor, epsilon: float, iterations: int) -> Tensor:
    """Return a transport plan by differentiably unrolling Sinkhorn updates."""
    log_plan = -cost / epsilon
    for _ in range(iterations):
        log_plan = log_plan - torch.logsumexp(log_plan, dim=2, keepdim=True)
        log_plan = log_plan - torch.logsumexp(log_plan, dim=1, keepdim=True)
    # Average last step to reduce bias from the last row normalization
    log_plan_bis = log_plan - torch.logsumexp(log_plan, dim=2, keepdim=True)
    log_plan = 0.5 * (log_plan + log_plan_bis)
    return log_plan.exp()

def sinkhorn_unrolling_dual(cost: Tensor, epsilon: float, iterations: int) -> Tensor:
    """Return a transport plan by differentiably unrolling Sinkhorn updates in the dual."""
    log_plan = -cost / epsilon
    u = torch.zeros((log_plan.shape[0], log_plan.shape[1]), dtype=log_plan.dtype, device=log_plan.device)
    v = torch.zeros((log_plan.shape[0], log_plan.shape[2]), dtype=log_plan.dtype, device=log_plan.device)
    for _ in range(iterations):
        u = - torch.logsumexp(log_plan + v[:, None, :], dim=2)
        v = - torch.logsumexp(log_plan + u[:, :, None], dim=1)
    # Average last step to reduce bias from the last row normalization
    u_extra = - torch.logsumexp(log_plan + v[:, None, :], dim=2)
    u = 0.5 * (u + u_extra)
    log_plan = log_plan + u[:, :, None] + v[:, None, :]
    return log_plan.exp()

def sinkhorn_tol(
    cost: Tensor,
    epsilon: float,
    max_iter: int,
    tol: float,
    check_convergence_every: int,
) -> Tensor:
    K = -cost / epsilon
    b, n, m = K.shape
    u = torch.zeros((b, n), dtype=K.dtype, device=K.device)  # u = torch.log(a) - torch.logsumexp(K, dim=2).squeeze()
    v = torch.zeros((b, m), dtype=K.dtype, device=K.device)  # v = torch.log(b) - torch.logsumexp(K, dim=1).squeeze()
    active = torch.ones(b, dtype=torch.bool, device=K.device)
    for n_iters in range(max_iter):
        updated_u = - torch.logsumexp(K + v[:, None, :], dim=2)
        updated_v = - torch.logsumexp(K + updated_u[:, :, None], dim=1)
        u = torch.where(active[:, None], updated_u, u)
        v = torch.where(active[:, None], updated_v, v)
        if n_iters % check_convergence_every == 0:
            T = torch.exp(K + u[:, :, None] + v[:, None, :])
            row_error = torch.abs(torch.sum(T, dim=2) - 1).amax(dim=1)
            active = active & (row_error >= tol)
    # average last step so that both marginals are approximately correct
    u_extra = - torch.logsumexp(K + v[:, None, :], dim=2)
    u = 0.5 * (u + u_extra)
    log_plan = K + u[:, :, None] + v[:, None, :]
    return log_plan.exp()

class SinkhornImplicit(torch.autograd.Function):
    """
    An implementation of a Sinkhorn layer with our custom backward module, based on implicit differentiation
    :param cost: input cost matrix, size [*,m,n], where * are arbitrarily many batch dimensions
    :param epsilon: entropy regularization weight
    :param max_iter: maximum number of Sinkhorn iterations
    :param tol: tolerance for convergence
    :param check_convergence_every: check convergence every n iterations
    :return: optimized soft permutation matrix
    """
    @staticmethod
    def forward(ctx, cost, epsilon, max_iter=10000,tol=1e-5,check_convergence_every=10):
        p = sinkhorn_tol(cost, epsilon=epsilon, max_iter=max_iter, tol=tol, check_convergence_every=check_convergence_every)
        ctx.save_for_backward(p, torch.sum(p, dim=-1), torch.sum(p, dim=-2))
        ctx.lambd_sink = epsilon
        return p

    @staticmethod
    def backward(ctx, grad_p):
        p, a, b = ctx.saved_tensors
        *batch_shape, m, n = p.shape          # unpack only the last two dims as (m, n)

        grad_p = grad_p * (-1 / ctx.lambd_sink) * p

        K = torch.cat((
            torch.cat((torch.diag_embed(a), p), dim=-1),
            torch.cat((p.transpose(-2, -1), torch.diag_embed(b)), dim=-1),
        ), dim=-2)[..., :-1, :-1]

        t = torch.cat((
            grad_p.sum(dim=-1),
            grad_p[..., :, :-1].sum(dim=-2),
        ), dim=-1)                             # shape (*batch, m+n-1)

        eye = torch.eye(K.shape[-1], device=K.device, dtype=K.dtype)
        K = K + 1e-7 * eye
        grad_ab, _ = torch.linalg.solve_ex(K, t)     # solve handles batching fine

        grad_a = grad_ab[..., :m]              # shape (*batch, m)
        zero_pad = torch.zeros(*batch_shape, 1, device=grad_p.device, dtype=grad_p.dtype)
        grad_b = torch.cat((grad_ab[..., m:], zero_pad), dim=-1)   # shape (*batch, n)

        U = grad_a.unsqueeze(-1) + grad_b.unsqueeze(-2)   # (*batch, m, 1) + (*batch, 1, n) -> (*batch, m, n)
        grad_p = grad_p - p * U

        grad_a = -ctx.lambd_sink * grad_a
        grad_b = -ctx.lambd_sink * grad_b

        return grad_p, None, None, None, None


class BaseMatcher(nn.Module):
    """Common validation, padding replacement, and Hungarian projection."""

    def __init__(
        self,
        d_node_decoder: int,
        n_nodes: int,
        normalize_cost: bool,
        objective_parameters: ObjectiveParameters | None = None,
    ) -> None:
        super().__init__()
        if d_node_decoder < 1 or n_nodes < 1:
            raise ValueError("d_node_decoder and n_nodes must be positive.")
        self.d_node_decoder = d_node_decoder
        self.n_nodes = n_nodes
        self.normalize_cost = normalize_cost
        objective = objective_parameters or ObjectiveParameters()
        self.feature_diffusion = objective.feature_diffusion
        self.alpha_feature_diffusion = objective.alpha_feature_diffusion
        self.target_padding_embedding = nn.Parameter(
            torch.empty(1, 1, d_node_decoder)
        )
        nn.init.xavier_uniform_(self.target_padding_embedding)

    def _validate(
        self,
        predicted_node_embeddings: Tensor,
        target_node_embeddings: Tensor,
        target_padding_mask: Tensor,
    ) -> None:
        expected = (
            predicted_node_embeddings.shape[0],
            self.n_nodes,
            self.d_node_decoder,
        )
        if predicted_node_embeddings.shape != expected:
            raise ValueError(f"Predicted node embeddings must have shape {expected}.")
        if target_node_embeddings.shape != expected:
            raise ValueError(f"Target node embeddings must have shape {expected}.")
        if (
            target_padding_mask.shape != expected[:2]
            or target_padding_mask.dtype != torch.bool
        ):
            raise ValueError("target_padding_mask must be bool [batch, target].")

    def _replace_target_padding(
        self, target_node_embeddings: Tensor, target_padding_mask: Tensor
    ) -> Tensor:
        padding = self.target_padding_embedding.expand_as(target_node_embeddings)
        return torch.where(
            target_padding_mask.unsqueeze(-1), padding, target_node_embeddings
        )

    def _normalize(self, cost: Tensor) -> Tensor:
        if not self.normalize_cost:
            return cost
        floor = torch.finfo(cost.dtype).eps
        denominator = cost.sum(dim=(1, 2), keepdim=True).clamp_min(floor)
        return cost / denominator

    def _add_feature_diffusion_cost(
        self,
        cost: Tensor,
        *,
        predicted_diffused_features: Tensor | None,
        target_diffused_features: Tensor | None,
        target_padding_mask: Tensor,
    ) -> Tensor:
        """Add the original Any2Graph squared-L2 ``A@F`` linear cost."""

        if not self.feature_diffusion:
            return cost
        if predicted_diffused_features is None or target_diffused_features is None:
            raise ValueError(
                "Feature diffusion requires predicted and target diffused features."
            )
        expected = (cost.shape[0], self.n_nodes)
        if predicted_diffused_features.shape[:2] != expected:
            raise ValueError(
                "Predicted diffused features must have shape "
                "[batch, predicted, features]."
            )
        if target_diffused_features.shape[:2] != expected:
            raise ValueError(
                "Target diffused features must have shape [batch, target, features]."
            )
        real_targets = (~target_padding_mask).to(cost.dtype)
        target_sizes = real_targets.sum(dim=1, keepdim=True).clamp_min(1.0)
        target_weights = self.n_nodes * real_targets / target_sizes
        diffused_cost = pairwise_squared_l2(
            predicted_diffused_features, target_diffused_features
        )
        return cost + (
            self.alpha_feature_diffusion
            * diffused_cost
            * target_weights.unsqueeze(1)
        )

    def cost_matrix(
        self,
        *,
        predicted_node_embeddings: Tensor,
        target_node_embeddings: Tensor,
        target_padding_mask: Tensor,
        predicted_diffused_features: Tensor | None = None,
        target_diffused_features: Tensor | None = None,
    ) -> Tensor:
        raise NotImplementedError

    def hard_match(
        self,
        *,
        predicted_node_embeddings: Tensor,
        target_node_embeddings: Tensor,
        target_padding_mask: Tensor,
        predicted_diffused_features: Tensor | None = None,
        target_diffused_features: Tensor | None = None,
    ) -> HardMatchResult:
        """Project the learned cost to a target-to-predicted Hungarian matching."""
        cost = self.cost_matrix(
            predicted_node_embeddings=predicted_node_embeddings,
            target_node_embeddings=target_node_embeddings,
            target_padding_mask=target_padding_mask,
            predicted_diffused_features=predicted_diffused_features,
            target_diffused_features=target_diffused_features,
        )
        permutations = []
        for batch_cost in cost.detach().cpu().numpy():
            _, predicted_for_target = linear_sum_assignment(batch_cost.T)
            permutations.append(predicted_for_target)
        permutation_tensor = torch.from_numpy(np.stack(permutations)).to(
            device=cost.device, dtype=torch.long
        )
        return HardMatchResult(
            permutations=permutation_tensor,
            matrices=permutations_to_matrices(permutation_tensor).to(cost.dtype),
            cost=cost,
        )


class SinkhornMatcher(BaseMatcher):
    """Learned projected L1 costs followed by fixed-step log Sinkhorn."""

    def __init__(
        self,
        d_node_decoder: int,
        n_nodes: int,
        parameters: MatcherParameters | None = None,
        objective_parameters: ObjectiveParameters | None = None,
    ) -> None:
        configuration = parameters or MatcherParameters()
        super().__init__(
            d_node_decoder,
            n_nodes,
            configuration.normalize_matcher_cost,
            objective_parameters,
        )
        self.configuration = configuration
        self.predicted_positions = nn.Parameter(
            torch.empty(1, n_nodes, d_node_decoder)
        )
        self.predicted_projection = nn.Linear(d_node_decoder, configuration.matcher_dim)
        self.target_projection = nn.Linear(d_node_decoder, configuration.matcher_dim)
        nn.init.xavier_uniform_(self.predicted_positions)

    def cost_matrix(
        self,
        *,
        predicted_node_embeddings: Tensor,
        target_node_embeddings: Tensor,
        target_padding_mask: Tensor,
        predicted_diffused_features: Tensor | None = None,
        target_diffused_features: Tensor | None = None,
    ) -> Tensor:
        self._validate(
            predicted_node_embeddings, target_node_embeddings, target_padding_mask
        )
        targets = self._replace_target_padding(
            target_node_embeddings, target_padding_mask
        )
        predicted = predicted_node_embeddings + self.predicted_positions
        predicted = self.predicted_projection(predicted)
        targets = self.target_projection(targets)
        cost = torch.cdist(predicted, targets, p=1)
        cost = self._add_feature_diffusion_cost(
            cost,
            predicted_diffused_features=predicted_diffused_features,
            target_diffused_features=target_diffused_features,
            target_padding_mask=target_padding_mask,
        )
        return self._normalize(cost)

    def forward(
        self,
        *,
        predicted_node_embeddings: Tensor,
        target_node_embeddings: Tensor,
        target_padding_mask: Tensor,
        predicted_diffused_features: Tensor | None = None,
        target_diffused_features: Tensor | None = None,
    ) -> MatchResult:
        """Return an approximately bistochastic ``T[p,t]`` transport plan."""
        cost = self.cost_matrix(
            predicted_node_embeddings=predicted_node_embeddings,
            target_node_embeddings=target_node_embeddings,
            target_padding_mask=target_padding_mask,
            predicted_diffused_features=predicted_diffused_features,
            target_diffused_features=target_diffused_features,
        )
        if self.configuration.sinkhorn_mode == "unrolling":
            plan = sinkhorn_unrolling_dual(
                cost,
                self.configuration.matcher_epsilon,
                self.configuration.sinkhorn_iterations
            )
        else:
            plan = SinkhornImplicit.apply(
                cost,
                self.configuration.matcher_epsilon,
                self.configuration.sinkhorn_iterations,
                self.configuration.sinkhorn_tolerance,
                self.configuration.sinkhorn_check_convergence_every
            )
        diagnostics = marginal_diagnostics(plan)
        diagnostics["sinkhorn_iterations"] = plan.new_tensor(
            self.configuration.sinkhorn_iterations
        )
        return MatchResult(plan=plan, cost=cost, diagnostics=diagnostics)


class SoftsortMatcher(BaseMatcher):
    """Reference-compatible target-score sorting with one-sided soft marginals."""

    def __init__(
        self,
        d_node_decoder: int,
        n_nodes: int,
        parameters: MatcherParameters | None = None,
        objective_parameters: ObjectiveParameters | None = None,
    ) -> None:
        configuration = parameters or MatcherParameters(matcher_type="softsort")
        super().__init__(
            d_node_decoder,
            n_nodes,
            configuration.normalize_matcher_cost,
            objective_parameters,
        )
        self.configuration = configuration
        self.target_score = nn.Linear(d_node_decoder, 1)

    def cost_matrix(
        self,
        *,
        predicted_node_embeddings: Tensor,
        target_node_embeddings: Tensor,
        target_padding_mask: Tensor,
        predicted_diffused_features: Tensor | None = None,
        target_diffused_features: Tensor | None = None,
    ) -> Tensor:
        self._validate(
            predicted_node_embeddings, target_node_embeddings, target_padding_mask
        )
        targets = self._replace_target_padding(
            target_node_embeddings, target_padding_mask
        )
        scores = self.target_score(targets).squeeze(-1)
        sorted_scores = scores.sort(dim=1).values
        cost = (sorted_scores.unsqueeze(2) - scores.unsqueeze(1)).abs()
        dependency = predicted_node_embeddings.sum(dim=(1, 2), keepdim=True) * 0.0
        cost = self._add_feature_diffusion_cost(
            cost + dependency,
            predicted_diffused_features=predicted_diffused_features,
            target_diffused_features=target_diffused_features,
            target_padding_mask=target_padding_mask,
        )
        return self._normalize(cost)

    def forward(
        self,
        *,
        predicted_node_embeddings: Tensor,
        target_node_embeddings: Tensor,
        target_padding_mask: Tensor,
        predicted_diffused_features: Tensor | None = None,
        target_diffused_features: Tensor | None = None,
    ) -> MatchResult:
        """Return Softsort's one-sided normalized ``T[p,t]`` transport plan."""
        cost = self.cost_matrix(
            predicted_node_embeddings=predicted_node_embeddings,
            target_node_embeddings=target_node_embeddings,
            target_padding_mask=target_padding_mask,
            predicted_diffused_features=predicted_diffused_features,
            target_diffused_features=target_diffused_features,
        )
        plan = torch.softmax(-cost / self.configuration.matcher_epsilon, dim=1)
        return MatchResult(
            plan=plan, cost=cost, diagnostics=marginal_diagnostics(plan)
        )


def build_matcher(
    d_node_decoder: int,
    n_nodes: int,
    parameters: MatcherParameters | None = None,
    objective_parameters: ObjectiveParameters | None = None,
) -> BaseMatcher:
    """Instantiate the configured Sinkhorn or Softsort matcher."""
    configuration = parameters or MatcherParameters()
    if configuration.matcher_type == "sinkhorn":
        return SinkhornMatcher(
            d_node_decoder, n_nodes, configuration, objective_parameters
        )
    return SoftsortMatcher(
        d_node_decoder, n_nodes, configuration, objective_parameters
    )


if __name__ == "__main__":
    import torch

    torch.set_printoptions(precision=1, sci_mode=False)
    n = 200
    K = 1 - torch.eye(n) + 0.1 *torch.rand((n, n))
    K.requires_grad_(True)
    p_target = torch.eye(n)

    p_unrolling = sinkhorn_unrolling(
        K.unsqueeze(0), epsilon=0.1, iterations=1000
    ).squeeze(0)
    loss_unrolling = ((p_unrolling - p_target)**2).mean()
    grad_unrolling = torch.autograd.grad(loss_unrolling, K)[0]

    p_implicit = SinkhornImplicit.apply(K.unsqueeze(0), 0.1, 1000, 1e-6, 10).squeeze(0)
    loss_implicit = ((p_implicit - p_target)**2).mean()
    grad_implicit = torch.autograd.grad(loss_implicit, K)[0]

    print(torch.allclose(grad_unrolling, grad_implicit, atol=1e-5))
    print(torch.allclose(loss_unrolling, loss_implicit, atol=1e-5))
