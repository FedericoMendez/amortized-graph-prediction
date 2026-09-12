"""Molecular port of GRALE's corrected direct-plan graph objective."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as functional
from torch import Tensor, nn
from torch.special import entr

from any2graph_v2.data.dense_data import BatchedDenseData
from any2graph_v2.parameter import ObjectiveParameters

from .feature_diffusion import one_hop_feature_diffusion, pairwise_squared_l2
from .regularization import MarginalKL, _validate_plan


class LinearLoss(nn.Module):
    """GRALE linear loss ``sum_pt T_pt L(F1_p, F2_t)``."""

    def pairwise_loss(self, predicted: Tensor, target: Tensor) -> Tensor:
        raise NotImplementedError

    def forward(
        self,
        plan: Tensor,
        predicted: Tensor,
        target: Tensor,
        target_weight: Tensor | None = None,
    ) -> Tensor:
        _validate_plan(plan)
        cost = self.pairwise_loss(predicted, target)
        if cost.shape != plan.shape:
            raise ValueError("The linear pairwise cost must have the plan shape.")
        if target_weight is not None:
            if target_weight.shape != (plan.shape[0], plan.shape[2]):
                raise ValueError("target_weight must be [batch, target].")
            cost = cost * target_weight.unsqueeze(1)
        return (plan * cost).sum(dim=(1, 2))


class LinearBCE(LinearLoss):
    """GRALE pairwise binary cross-entropy for node presence."""

    def pairwise_loss(self, predicted_logits: Tensor, target_values: Tensor) -> Tensor:
        batch, predicted_size = predicted_logits.shape
        expected_target = (batch, target_values.shape[1])
        if target_values.shape != expected_target:
            raise ValueError("target_values must have shape [batch, target].")
        return functional.binary_cross_entropy_with_logits(
            predicted_logits.unsqueeze(2).expand(-1, -1, target_values.shape[1]),
            target_values.unsqueeze(1).expand(-1, predicted_size, -1),
            reduction="none",
        )


class LinearCE(LinearLoss):
    """GRALE pairwise categorical cross-entropy for node labels."""

    def pairwise_loss(
        self, predicted_logits: Tensor, target_probabilities: Tensor
    ) -> Tensor:
        if predicted_logits.ndim != 3 or target_probabilities.ndim != 3:
            raise ValueError("categorical node tensors must have three dimensions.")
        if predicted_logits.shape[0] != target_probabilities.shape[0]:
            raise ValueError("predicted and target batch sizes must agree.")
        if predicted_logits.shape[2] != target_probabilities.shape[2]:
            raise ValueError("predicted and target class counts must agree.")
        return -torch.bmm(
            functional.log_softmax(predicted_logits, dim=-1),
            target_probabilities.transpose(1, 2),
        )


class LinearSquaredL2(LinearLoss):
    """Any2Graph pairwise squared-L2 loss for continuous node features."""

    def pairwise_loss(self, predicted: Tensor, target: Tensor) -> Tensor:
        return pairwise_squared_l2(predicted, target)


class QuadraticLoss(nn.Module):
    """GRALE quadratic loss with its factorized tensor-product evaluation."""

    def __init__(self, exclude_self_loops: bool = False) -> None:
        super().__init__()
        self.exclude_self_loops = exclude_self_loops

    def f1(self, predicted: Tensor) -> Tensor:
        raise NotImplementedError

    def f2(self, target: Tensor) -> Tensor:
        raise NotImplementedError

    def h1(self, predicted: Tensor) -> Tensor:
        raise NotImplementedError

    def h2(self, target: Tensor) -> Tensor:
        raise NotImplementedError

    def forward(
        self,
        plan: Tensor,
        predicted: Tensor,
        target: Tensor,
        target_node_weight: Tensor | None = None,
    ) -> Tensor:
        _validate_plan(plan)
        batch, predicted_size, target_size = plan.shape
        if predicted.shape[:3] != (batch, predicted_size, predicted_size):
            raise ValueError("predicted edge tensor has incompatible axes.")
        if target.shape[:3] != (batch, target_size, target_size):
            raise ValueError("target edge tensor has incompatible axes.")
        if target_node_weight is None:
            target_node_weight = plan.new_ones(batch, target_size)
        elif target_node_weight.shape != (batch, target_size):
            raise ValueError("target_node_weight must be [batch, target].")

        predicted_weight = plan.new_ones(batch, predicted_size)
        predicted_pair_weight = (
            predicted_weight.unsqueeze(2) * predicted_weight.unsqueeze(1)
        )
        target_pair_weight = (
            target_node_weight.unsqueeze(2) * target_node_weight.unsqueeze(1)
        )
        if self.exclude_self_loops:
            predicted_pair_weight = predicted_pair_weight * (
                1.0 - torch.eye(predicted_size, device=plan.device, dtype=plan.dtype)
            )
            target_pair_weight = target_pair_weight * (
                1.0 - torch.eye(target_size, device=plan.device, dtype=plan.dtype)
            )

        f1 = self.f1(predicted) * predicted_pair_weight
        f2 = self.f2(target) * target_pair_weight
        h1 = self.h1(predicted) * predicted_pair_weight.unsqueeze(-1)
        h2 = self.h2(target) * target_pair_weight.unsqueeze(-1)

        # Exact port of GRALE's decomposition
        # L(a,b) = f1(a) + f2(b) - <h1(a), h2(b)>.
        term_a = torch.bmm(f1, torch.bmm(plan, target_pair_weight.transpose(1, 2)))
        term_b = torch.bmm(
            predicted_pair_weight, torch.bmm(plan, f2.transpose(1, 2))
        )
        term_c = torch.einsum("bpqd,bqt,butd->bpu", h1, plan, h2)
        pairwise = term_a + term_b - term_c
        return (pairwise * plan).sum(dim=(1, 2))


class QuadraticBCE(QuadraticLoss):
    """GRALE quadratic BCE for adjacency logits and binary targets."""

    def f1(self, predicted_logits: Tensor) -> Tensor:
        return -functional.logsigmoid(predicted_logits)

    def f2(self, target_values: Tensor) -> Tensor:
        return -entr(target_values) - entr(1.0 - target_values)

    def h1(self, predicted_logits: Tensor) -> Tensor:
        return -predicted_logits.unsqueeze(-1)

    def h2(self, target_values: Tensor) -> Tensor:
        return (1.0 - target_values).unsqueeze(-1)


class QuadraticCE(QuadraticLoss):
    """GRALE quadratic categorical CE/KL for edge labels."""

    def f1(self, predicted_logits: Tensor) -> Tensor:
        return torch.logsumexp(predicted_logits, dim=-1)

    def f2(self, target_probabilities: Tensor) -> Tensor:
        return -entr(target_probabilities).sum(dim=-1)

    def h1(self, predicted_logits: Tensor) -> Tensor:
        return predicted_logits

    def h2(self, target_probabilities: Tensor) -> Tensor:
        return target_probabilities


@dataclass
class ReconstructionLossResult:
    """Scalar training loss plus raw per-component diagnostics."""

    loss: Tensor
    per_graph_loss: Tensor
    components: dict[str, Tensor]
    per_graph_components: dict[str, Tensor]
    diagnostics: dict[str, Tensor]


class GraphReconstructionObjective(nn.Module):
    """Corrected GRALE direct-plan objective for molecular graph fields."""

    def __init__(self, parameters: ObjectiveParameters | None = None) -> None:
        super().__init__()
        self.configuration = parameters or ObjectiveParameters()
        self.presence_objective = LinearBCE()
        self.atom_objective = LinearCE()
        self.feature_diffusion_objective = LinearSquaredL2()
        self.bond_objective = QuadraticCE(self.configuration.exclude_self_loops)
        self.adjacency_objective = QuadraticBCE(
            self.configuration.exclude_self_loops
        )
        self.marginal_objective = MarginalKL()

    def _validate(
        self, predicted_graph: BatchedDenseData, targets: BatchedDenseData, plan: Tensor
    ) -> None:
        _validate_plan(plan)
        if targets.h.dtype != torch.bool:
            raise TypeError("targets.h must be boolean.")
        if not predicted_graph.h.is_floating_point():
            raise TypeError("predicted_graph.h must contain floating logits.")
        expected = (targets.batchsize, predicted_graph.size, targets.size)
        if plan.shape != expected:
            raise ValueError(f"plan must have shape {expected}.")
        if predicted_graph.batchsize != targets.batchsize:
            raise ValueError("predicted and target batch sizes must agree.")
        for graph, role in ((predicted_graph, "predicted"), (targets, "target")):
            if "labels" not in graph.nodes or "labels" not in graph.edges:
                raise ValueError(f"{role} graph requires node and edge labels.")
            if "adjacency" not in graph.edges:
                raise ValueError(f"{role} graph requires adjacency.")
        if predicted_graph.nodes.labels.shape[-1] != targets.nodes.labels.shape[-1]:
            raise ValueError("predicted and target atom class counts must agree.")
        if self.configuration.feature_diffusion:
            if "diffused_labels" not in predicted_graph.nodes:
                raise ValueError(
                    "Feature diffusion requires predicted diffused_labels."
                )
            if (
                predicted_graph.nodes.diffused_labels.shape
                != predicted_graph.nodes.labels.shape
            ):
                raise ValueError(
                    "Predicted diffused_labels must match node-label shape."
                )
        if predicted_graph.edges.labels.shape[-1] != targets.edges.labels.shape[-1]:
            raise ValueError("predicted and target bond class counts must agree.")
        if torch.any(targets.h.sum(dim=1) == 0):
            raise ValueError("Every target graph must contain at least one real node.")

    def forward(
        self,
        *,
        predicted_graph: BatchedDenseData,
        targets: BatchedDenseData,
        plan: Tensor,
    ) -> ReconstructionLossResult:
        """Evaluate all weighted GRALE loss terms using the supplied ``T[p,t]``.

        The plan is used directly for presence, atom, bond, adjacency, and
        marginal terms. This method deliberately does not apply a hard
        permutation: Hungarian alignment belongs only to validation metrics.
        """
        self._validate(predicted_graph, targets, plan)
        target_mask = targets.h.to(plan.dtype)
        target_sizes = target_mask.sum(dim=1)

        presence = self.presence_objective(
            plan, predicted_graph.h, target_mask
        ) / targets.size
        atom = self.atom_objective(
            plan, predicted_graph.nodes.labels, targets.nodes.labels, target_mask
        ) / target_sizes
        feature_diffusion = None
        if self.configuration.feature_diffusion:
            feature_diffusion = self.feature_diffusion_objective(
                plan,
                predicted_graph.nodes.diffused_labels,
                one_hop_feature_diffusion(targets),
                target_mask,
            ) / target_sizes
        bond, adjacency = self._edge_components(
            predicted_graph=predicted_graph,
            targets=targets,
            plan=plan,
            target_mask=target_mask,
            target_sizes=target_sizes,
        )
        marginal = self.marginal_objective(plan)

        per_graph_components = {
            "presence": presence,
            "atom": atom,
            "bond": bond,
            "adjacency": adjacency,
            "marginal": marginal,
        }
        if feature_diffusion is not None:
            per_graph_components["feature_diffusion"] = feature_diffusion
        return self._build_result(
            per_graph_components=per_graph_components,
            plan=plan,
            target_sizes=target_sizes,
        )

    def _build_result(
        self,
        *,
        per_graph_components: dict[str, Tensor],
        plan: Tensor,
        target_sizes: Tensor,
    ) -> ReconstructionLossResult:
        """Apply configured weights and attach shared plan diagnostics."""

        weights = {
            "presence": self.configuration.alpha_presence,
            "atom": self.configuration.alpha_atom,
            "bond": self.configuration.alpha_bond,
            "adjacency": self.configuration.alpha_adjacency,
            "marginal": self.configuration.alpha_marginal,
            "feature_diffusion": self.configuration.alpha_feature_diffusion,
        }
        per_graph_loss = sum(
            weights[name] * value for name, value in per_graph_components.items()
        )
        components = {
            name: value.mean() for name, value in per_graph_components.items()
        }
        row_error = (plan.sum(dim=2) - 1.0).abs()
        column_error = (plan.sum(dim=1) - 1.0).abs()
        diagnostics = {
            "row_marginal_max_error": row_error.amax(),
            "row_marginal_mean_error": row_error.mean(),
            "column_marginal_max_error": column_error.amax(),
            "column_marginal_mean_error": column_error.mean(),
            "target_size_mean": target_sizes.mean(),
            "target_size_min": target_sizes.amin(),
            "target_size_max": target_sizes.amax(),
        }
        return ReconstructionLossResult(
            loss=per_graph_loss.mean(),
            per_graph_loss=per_graph_loss,
            components=components,
            per_graph_components=per_graph_components,
            diagnostics=diagnostics,
        )

    def _edge_components(
        self,
        *,
        predicted_graph: BatchedDenseData,
        targets: BatchedDenseData,
        plan: Tensor,
        target_mask: Tensor,
        target_sizes: Tensor,
    ) -> tuple[Tensor, Tensor]:
        """Return categorical-bond and binary-adjacency losses per graph."""

        bond = self.bond_objective(
            plan,
            predicted_graph.edges.labels,
            targets.edges.labels,
            target_mask,
        ) / target_sizes.square()
        adjacency = self.adjacency_objective(
            plan,
            predicted_graph.edges.adjacency,
            targets.edges.adjacency,
            target_mask,
        ) / target_sizes.square()
        return bond, adjacency
