"""Half- and fully-aligned relaxations of the graph edge objective."""

from __future__ import annotations

import torch
import torch.nn.functional as functional
from torch import Tensor

from any2graph_v2.data.dense_data import BatchedDenseData
from any2graph_v2.parameter import ObjectiveParameters

from .feature_diffusion import one_hop_feature_diffusion
from .graph_reconstruction import (
    GraphReconstructionObjective,
    ReconstructionLossResult,
)


def _normalize_plan(plan: Tensor, dimension: int) -> Tensor:
    """Normalize one plan marginal without hiding errors from MarginalKL."""

    denominator = plan.sum(dim=dimension, keepdim=True)
    tiny = torch.finfo(plan.dtype).tiny
    return torch.where(
        denominator > 0,
        plan / denominator.clamp_min(tiny),
        torch.zeros_like(plan),
    )


def _conditional(numerator: Tensor, weight: Tensor) -> Tensor:
    """Form a conditional target distribution at positions with positive mass."""

    tiny = torch.finfo(numerator.dtype).tiny
    expanded_weight = weight
    while expanded_weight.ndim < numerator.ndim:
        expanded_weight = expanded_weight.unsqueeze(-1)
    return torch.where(
        expanded_weight > 0,
        numerator / expanded_weight.clamp_min(tiny),
        torch.zeros_like(numerator),
    )


def _categorical_kl(predicted: Tensor, target: Tensor) -> Tensor:
    """Return pointwise ``KL(target || predicted)`` for probabilities."""

    epsilon = torch.finfo(predicted.dtype).eps
    predicted = predicted.clamp_min(epsilon)
    target = target.clamp_min(0.0)
    return (
        target
        * (target.clamp_min(epsilon).log() - predicted.log())
    ).sum(dim=-1)


def _bernoulli_kl(predicted: Tensor, target: Tensor) -> Tensor:
    """Return pointwise Bernoulli ``KL(target || predicted)``."""

    epsilon = torch.finfo(predicted.dtype).eps
    predicted = predicted.clamp(epsilon, 1.0 - epsilon)
    target = target.clamp(0.0, 1.0)
    return (
        target
        * (target.clamp_min(epsilon).log() - predicted.log())
        + (1.0 - target)
        * (
            (1.0 - target).clamp_min(epsilon).log()
            - (1.0 - predicted).log()
        )
    )


def _bernoulli_cross_entropy(predicted: Tensor, target: Tensor) -> Tensor:
    """Return pointwise Bernoulli cross-entropy for probabilities."""

    epsilon = torch.finfo(predicted.dtype).eps
    predicted = predicted.clamp(epsilon, 1.0 - epsilon)
    target = target.clamp(0.0, 1.0)
    return -(
        target * predicted.log()
        + (1.0 - target) * (1.0 - predicted).log()
    )


def _masked_target_edges(
    targets: BatchedDenseData, target_mask: Tensor
) -> tuple[Tensor, Tensor]:
    pair_mask = target_mask.unsqueeze(2) * target_mask.unsqueeze(1)
    adjacency = targets.edges.adjacency.to(target_mask.dtype) * pair_mask
    labels = targets.edges.labels.to(target_mask.dtype) * pair_mask.unsqueeze(-1)
    return labels, adjacency


class GraphReconstructionObjective_alt_a(GraphReconstructionObjective):
    """Paper relaxation ``J_a`` comparing ``A @ T`` with ``T @ B``.

    Node, presence, marginal, and optional feature-diffusion components retain
    the direct-plan objective. Only bond and adjacency reconstruction use the
    half-aligned relaxation.
    """

    def __init__(self, parameters: ObjectiveParameters | None = None) -> None:
        super().__init__(parameters)

    def _edge_components(
        self,
        *,
        predicted_graph: BatchedDenseData,
        targets: BatchedDenseData,
        plan: Tensor,
        target_mask: Tensor,
        target_sizes: Tensor,
    ) -> tuple[Tensor, Tensor]:
        row_plan = _normalize_plan(plan, dimension=2)
        column_plan = _normalize_plan(plan, dimension=1)
        target_labels, target_adjacency = _masked_target_edges(
            targets, target_mask
        )

        predicted_labels = functional.softmax(
            predicted_graph.edges.labels, dim=-1
        )
        predicted_adjacency = torch.sigmoid(predicted_graph.edges.adjacency)
        half_predicted_labels = torch.einsum(
            "bpqc,bqt->bptc", predicted_labels, column_plan
        )
        half_predicted_adjacency = torch.einsum(
            "bpq,bqt->bpt", predicted_adjacency, column_plan
        )

        half_target_labels_numerator = torch.einsum(
            "bpt,btuc->bpuc", row_plan, target_labels
        )
        half_target_adjacency_numerator = torch.einsum(
            "bpt,btu->bpu", row_plan, target_adjacency
        )
        transported_real_mass = torch.einsum(
            "bpt,bt->bp", row_plan, target_mask
        )
        pair_weight = transported_real_mass.unsqueeze(2) * target_mask.unsqueeze(1)
        half_target_labels = _conditional(
            half_target_labels_numerator, pair_weight
        )
        half_target_adjacency = _conditional(
            half_target_adjacency_numerator, pair_weight
        )

        if self.configuration.exclude_self_loops:
            # For a hard permutation, p and t denote the same node precisely
            # where T[p,t] == 1. This is the cross-order diagonal of J_a.
            pair_weight = pair_weight * (1.0 - row_plan)

        normalizer = target_sizes.square()
        bond = (
            _categorical_kl(half_predicted_labels, half_target_labels)
            * pair_weight
        ).sum(dim=(1, 2)) / normalizer
        adjacency = (
            _bernoulli_kl(half_predicted_adjacency, half_target_adjacency)
            * pair_weight
        ).sum(dim=(1, 2)) / normalizer
        return bond, adjacency


class GraphReconstructionObjective_alt_b(GraphReconstructionObjective):
    """Paper relaxation ``J_b`` comparing ``A`` with ``T @ B @ T.T``.

    Node, presence, marginal, and optional feature-diffusion components retain
    the direct-plan objective. Only bond and adjacency reconstruction use the
    fully target-aligned relaxation.
    """

    def __init__(self, parameters: ObjectiveParameters | None = None) -> None:
        super().__init__(parameters)

    def _edge_components(
        self,
        *,
        predicted_graph: BatchedDenseData,
        targets: BatchedDenseData,
        plan: Tensor,
        target_mask: Tensor,
        target_sizes: Tensor,
    ) -> tuple[Tensor, Tensor]:
        row_plan = _normalize_plan(plan, dimension=2)
        target_labels, target_adjacency = _masked_target_edges(
            targets, target_mask
        )

        aligned_target_labels_numerator = torch.einsum(
            "bpt,btuc,bqu->bpqc", row_plan, target_labels, row_plan
        )
        aligned_target_adjacency_numerator = torch.einsum(
            "bpt,btu,bqu->bpq", row_plan, target_adjacency, row_plan
        )
        transported_real_mass = torch.einsum(
            "bpt,bt->bp", row_plan, target_mask
        )
        pair_weight = (
            transported_real_mass.unsqueeze(2)
            * transported_real_mass.unsqueeze(1)
        )
        aligned_target_labels = _conditional(
            aligned_target_labels_numerator, pair_weight
        )
        aligned_target_adjacency = _conditional(
            aligned_target_adjacency_numerator, pair_weight
        )

        if self.configuration.exclude_self_loops:
            pair_weight = pair_weight * (
                1.0
                - torch.eye(
                    predicted_graph.size,
                    device=plan.device,
                    dtype=plan.dtype,
                )
            )

        predicted_labels = functional.softmax(
            predicted_graph.edges.labels, dim=-1
        )
        predicted_adjacency = torch.sigmoid(predicted_graph.edges.adjacency)
        normalizer = target_sizes.square()
        bond = (
            _categorical_kl(predicted_labels, aligned_target_labels)
            * pair_weight
        ).sum(dim=(1, 2)) / normalizer
        adjacency = (
            _bernoulli_kl(predicted_adjacency, aligned_target_adjacency)
            * pair_weight
        ).sum(dim=(1, 2)) / normalizer
        return bond, adjacency


class _PrimeGraphReconstructionObjective(GraphReconstructionObjective):
    """Shared directional node components for the reverse relaxations."""

    def _target_to_prediction_nodes(
        self,
        *,
        predicted_graph: BatchedDenseData,
        targets: BatchedDenseData,
        row_plan: Tensor,
        target_mask: Tensor,
        target_sizes: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor | None, Tensor]:
        transported_mass = torch.einsum("bpt,bt->bp", row_plan, target_mask)
        predicted_presence = torch.sigmoid(predicted_graph.h)
        presence = _bernoulli_cross_entropy(
            predicted_presence, transported_mass
        ).sum(dim=1) / targets.size

        target_atoms_numerator = torch.einsum(
            "bpt,btc->bpc",
            row_plan,
            targets.nodes.labels.to(row_plan.dtype) * target_mask.unsqueeze(-1),
        )
        target_atoms = _conditional(target_atoms_numerator, transported_mass)
        predicted_atoms = functional.softmax(
            predicted_graph.nodes.labels, dim=-1
        )
        atom = (
            _categorical_kl(predicted_atoms, target_atoms) * transported_mass
        ).sum(dim=1) / target_sizes

        feature_diffusion = None
        if self.configuration.feature_diffusion:
            target_diffusion_numerator = torch.einsum(
                "bpt,btc->bpc",
                row_plan,
                one_hop_feature_diffusion(targets) * target_mask.unsqueeze(-1),
            )
            target_diffusion = _conditional(
                target_diffusion_numerator, transported_mass
            )
            pointwise = (
                predicted_graph.nodes.diffused_labels - target_diffusion
            ).square().sum(dim=-1)
            feature_diffusion = (
                pointwise * transported_mass
            ).sum(dim=1) / target_sizes
        return presence, atom, feature_diffusion, transported_mass

    def _prediction_to_target_nodes(
        self,
        *,
        predicted_graph: BatchedDenseData,
        targets: BatchedDenseData,
        column_plan: Tensor,
        target_mask: Tensor,
        target_sizes: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor | None]:
        aligned_presence = torch.einsum(
            "bpt,bp->bt", column_plan, torch.sigmoid(predicted_graph.h)
        )
        presence = _bernoulli_cross_entropy(
            aligned_presence, target_mask
        ).sum(dim=1) / targets.size

        aligned_atoms = torch.einsum(
            "bpt,bpc->btc",
            column_plan,
            functional.softmax(predicted_graph.nodes.labels, dim=-1),
        )
        atom = (
            _categorical_kl(
                aligned_atoms, targets.nodes.labels.to(column_plan.dtype)
            )
            * target_mask
        ).sum(dim=1) / target_sizes

        feature_diffusion = None
        if self.configuration.feature_diffusion:
            aligned_diffusion = torch.einsum(
                "bpt,bpc->btc",
                column_plan,
                predicted_graph.nodes.diffused_labels,
            )
            pointwise = (
                aligned_diffusion - one_hop_feature_diffusion(targets)
            ).square().sum(dim=-1)
            feature_diffusion = (
                pointwise * target_mask
            ).sum(dim=1) / target_sizes
        return presence, atom, feature_diffusion

    def _prime_result(
        self,
        *,
        presence: Tensor,
        atom: Tensor,
        bond: Tensor,
        adjacency: Tensor,
        feature_diffusion: Tensor | None,
        plan: Tensor,
        target_sizes: Tensor,
    ) -> ReconstructionLossResult:
        per_graph_components = {
            "presence": presence,
            "atom": atom,
            "bond": bond,
            "adjacency": adjacency,
            "marginal": self.marginal_objective(plan),
        }
        if feature_diffusion is not None:
            per_graph_components["feature_diffusion"] = feature_diffusion
        return self._build_result(
            per_graph_components=per_graph_components,
            plan=plan,
            target_sizes=target_sizes,
        )


class GraphReconstructionObjective_alt_a_prime(_PrimeGraphReconstructionObjective):
    """Reverse half-aligned relaxation comparing ``T.T @ A`` and ``B @ T.T``."""

    def forward(
        self,
        *,
        predicted_graph: BatchedDenseData,
        targets: BatchedDenseData,
        plan: Tensor,
    ) -> ReconstructionLossResult:
        self._validate(predicted_graph, targets, plan)
        target_mask = targets.h.to(plan.dtype)
        target_sizes = target_mask.sum(dim=1)
        row_plan = _normalize_plan(plan, dimension=2)
        column_plan = _normalize_plan(plan, dimension=1)

        presence, atom, feature_diffusion, transported_mass = (
            self._target_to_prediction_nodes(
                predicted_graph=predicted_graph,
                targets=targets,
                row_plan=row_plan,
                target_mask=target_mask,
                target_sizes=target_sizes,
            )
        )
        target_labels, target_adjacency = _masked_target_edges(
            targets, target_mask
        )
        predicted_labels = functional.softmax(
            predicted_graph.edges.labels, dim=-1
        )
        predicted_adjacency = torch.sigmoid(predicted_graph.edges.adjacency)
        half_predicted_labels = torch.einsum(
            "bpt,bpqc->btqc", column_plan, predicted_labels
        )
        half_predicted_adjacency = torch.einsum(
            "bpt,bpq->btq", column_plan, predicted_adjacency
        )
        half_target_labels_numerator = torch.einsum(
            "btuc,bqu->btqc", target_labels, row_plan
        )
        half_target_adjacency_numerator = torch.einsum(
            "btu,bqu->btq", target_adjacency, row_plan
        )
        pair_weight = target_mask.unsqueeze(2) * transported_mass.unsqueeze(1)
        half_target_labels = _conditional(
            half_target_labels_numerator, pair_weight
        )
        half_target_adjacency = _conditional(
            half_target_adjacency_numerator, pair_weight
        )
        if self.configuration.exclude_self_loops:
            pair_weight = pair_weight * (1.0 - row_plan.transpose(1, 2))

        normalizer = target_sizes.square()
        bond = (
            _categorical_kl(half_predicted_labels, half_target_labels)
            * pair_weight
        ).sum(dim=(1, 2)) / normalizer
        adjacency = (
            _bernoulli_kl(half_predicted_adjacency, half_target_adjacency)
            * pair_weight
        ).sum(dim=(1, 2)) / normalizer
        return self._prime_result(
            presence=presence,
            atom=atom,
            bond=bond,
            adjacency=adjacency,
            feature_diffusion=feature_diffusion,
            plan=plan,
            target_sizes=target_sizes,
        )


class GraphReconstructionObjective_alt_b_prime(_PrimeGraphReconstructionObjective):
    """Reverse full relaxation comparing ``T.T @ A @ T`` with ``B``."""

    def forward(
        self,
        *,
        predicted_graph: BatchedDenseData,
        targets: BatchedDenseData,
        plan: Tensor,
    ) -> ReconstructionLossResult:
        self._validate(predicted_graph, targets, plan)
        target_mask = targets.h.to(plan.dtype)
        target_sizes = target_mask.sum(dim=1)
        column_plan = _normalize_plan(plan, dimension=1)

        presence, atom, feature_diffusion = self._prediction_to_target_nodes(
            predicted_graph=predicted_graph,
            targets=targets,
            column_plan=column_plan,
            target_mask=target_mask,
            target_sizes=target_sizes,
        )
        target_labels, target_adjacency = _masked_target_edges(
            targets, target_mask
        )
        aligned_predicted_labels = torch.einsum(
            "bpt,bpqc,bqu->btuc",
            column_plan,
            functional.softmax(predicted_graph.edges.labels, dim=-1),
            column_plan,
        )
        aligned_predicted_adjacency = torch.einsum(
            "bpt,bpq,bqu->btu",
            column_plan,
            torch.sigmoid(predicted_graph.edges.adjacency),
            column_plan,
        )
        pair_weight = target_mask.unsqueeze(2) * target_mask.unsqueeze(1)
        if self.configuration.exclude_self_loops:
            pair_weight = pair_weight * (
                1.0
                - torch.eye(
                    targets.size,
                    device=plan.device,
                    dtype=plan.dtype,
                )
            )

        normalizer = target_sizes.square()
        bond = (
            _categorical_kl(aligned_predicted_labels, target_labels)
            * pair_weight
        ).sum(dim=(1, 2)) / normalizer
        adjacency = (
            _bernoulli_kl(aligned_predicted_adjacency, target_adjacency)
            * pair_weight
        ).sum(dim=(1, 2)) / normalizer
        return self._prime_result(
            presence=presence,
            atom=atom,
            bond=bond,
            adjacency=adjacency,
            feature_diffusion=feature_diffusion,
            plan=plan,
            target_sizes=target_sizes,
        )


def build_graph_reconstruction_objective(
    parameters: ObjectiveParameters | None = None,
) -> GraphReconstructionObjective:
    """Construct the configured reconstruction objective."""

    configuration = parameters or ObjectiveParameters()
    objective_class = {
        "original": GraphReconstructionObjective,
        "alt_a": GraphReconstructionObjective_alt_a,
        "alt_b": GraphReconstructionObjective_alt_b,
        "alt_a_prime": GraphReconstructionObjective_alt_a_prime,
        "alt_b_prime": GraphReconstructionObjective_alt_b_prime,
    }[configuration.reconstruction_loss]
    return objective_class(configuration)
