"""CPU/GPU Frank--Wolfe graph matching adapted from Any2Graph PMFGW.

The legacy implementation alternates a linear assignment with an exact line
search (Frank--Wolfe/conditional gradient). This port retains that algorithm,
but constructs its costs from Any2GraphV2's atom, bond, adjacency, and presence
fields and returns the project-wide unit-marginal ``T[p,t]`` convention.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import torch
from scipy.optimize import linear_sum_assignment
from torch import Tensor, nn

from any2graph_v2.data.dense_data import BatchedDenseData
from any2graph_v2.losses import (
    GraphReconstructionObjective,
    QuadraticBCE,
    QuadraticCE,
    one_hop_feature_diffusion,
)
from any2graph_v2.parameter import ObjectiveParameters, SolverParameters


@dataclass
class SolverResult:
    """A solver plan and convergence information, all detached from autograd."""

    plan: Tensor
    iterations: Tensor


def _quadratic_tensor_product(
    objective: QuadraticBCE | QuadraticCE,
    predicted: Tensor,
    target: Tensor,
    target_node_weight: Tensor,
    plan: Tensor,
) -> Tensor:
    """Return the factorized pairwise edge cost ``K(plan)`` from the loss."""

    batch, predicted_size, target_size = plan.shape
    predicted_pair_weight = plan.new_ones(batch, predicted_size, predicted_size)
    target_pair_weight = (
        target_node_weight.unsqueeze(2) * target_node_weight.unsqueeze(1)
    )
    if objective.exclude_self_loops:
        predicted_pair_weight = predicted_pair_weight * (
            1.0
            - torch.eye(
                predicted_size, device=plan.device, dtype=plan.dtype
            ).unsqueeze(0)
        )
        target_pair_weight = target_pair_weight * (
            1.0
            - torch.eye(target_size, device=plan.device, dtype=plan.dtype).unsqueeze(0)
        )

    f1 = objective.f1(predicted) * predicted_pair_weight
    f2 = objective.f2(target) * target_pair_weight
    h1 = objective.h1(predicted) * predicted_pair_weight.unsqueeze(-1)
    h2 = objective.h2(target) * target_pair_weight.unsqueeze(-1)
    term_a = torch.bmm(f1, torch.bmm(plan, target_pair_weight.transpose(1, 2)))
    term_b = torch.bmm(
        predicted_pair_weight, torch.bmm(plan, f2.transpose(1, 2))
    )
    term_c = torch.einsum("bpqd,bqt,butd->bpu", h1, plan, h2)
    return term_a + term_b - term_c


class QuadraticGraphMatchingSolver(nn.Module):
    """Shared validation and objective construction for non-learned solvers."""

    def __init__(
        self,
        objective_parameters: ObjectiveParameters | None = None,
        solver_parameters: SolverParameters | None = None,
    ) -> None:
        super().__init__()
        self.configuration = objective_parameters or ObjectiveParameters()
        if self.configuration.reconstruction_loss != "original":
            raise ValueError(
                "Non-learned solvers optimize only reconstruction_loss=original."
            )
        self.solver_configuration = solver_parameters or SolverParameters()
        self._objective = GraphReconstructionObjective(self.configuration)

    def _validate(
        self, predicted_graph: BatchedDenseData, targets: BatchedDenseData
    ) -> None:
        size = predicted_graph.size
        placeholder = predicted_graph.h.new_ones(
            predicted_graph.batchsize, size, targets.size
        )
        self._objective._validate(predicted_graph, targets, placeholder)
        if size != targets.size:
            raise ValueError("Frank--Wolfe currently requires equal padded sizes.")

    def _cost_terms(
        self, predicted_graph: BatchedDenseData, targets: BatchedDenseData
    ) -> tuple[Tensor, Callable[[Tensor], Tensor]]:
        """Build the linear term and quadratic-cost operator on one device."""

        if self.backend == "gpu":
            if not predicted_graph.h.is_cuda:
                raise RuntimeError(
                    "solver_backend=gpu requires predictions and targets on CUDA."
                )
            compute_device = predicted_graph.h.device
        else:
            compute_device = torch.device("cpu")

        predicted = BatchedDenseData(
            h=predicted_graph.h.detach().to(
                device=compute_device, dtype=torch.float32
            ),
            nodes={
                name: value.detach().to(device=compute_device, dtype=torch.float32)
                for name, value in predicted_graph.nodes.items()
            },
            edges={
                name: value.detach().to(device=compute_device, dtype=torch.float32)
                for name, value in predicted_graph.edges.items()
            },
        )
        target = BatchedDenseData(
            h=targets.h.detach().to(device=compute_device),
            nodes={
                name: value.detach().to(device=compute_device, dtype=torch.float32)
                for name, value in targets.nodes.items()
            },
            edges={
                name: value.detach().to(device=compute_device, dtype=torch.float32)
                for name, value in targets.edges.items()
            },
        )
        dtype = predicted.h.dtype
        target_mask = target.h.to(dtype)
        target_sizes = target_mask.sum(dim=1)

        linear = (
            self.configuration.alpha_presence
            * self._objective.presence_objective.pairwise_loss(
                predicted.h, target_mask
            )
            / target.size
        )
        linear = linear + (
            self.configuration.alpha_atom
            * self._objective.atom_objective.pairwise_loss(
                predicted.nodes.labels, target.nodes.labels
            )
            * target_mask.unsqueeze(1)
            / target_sizes[:, None, None]
        )
        if self.configuration.feature_diffusion:
            linear = linear + (
                self.configuration.alpha_feature_diffusion
                * self._objective.feature_diffusion_objective.pairwise_loss(
                    predicted.nodes.diffused_labels,
                    one_hop_feature_diffusion(target),
                )
                * target_mask.unsqueeze(1)
                / target_sizes[:, None, None]
            )

        def quadratic(plan: Tensor) -> Tensor:
            scale = target_sizes.square()[:, None, None]
            bond = _quadratic_tensor_product(
                self._objective.bond_objective,
                predicted.edges.labels,
                target.edges.labels,
                target_mask,
                plan,
            )
            adjacency = _quadratic_tensor_product(
                self._objective.adjacency_objective,
                predicted.edges.adjacency,
                target.edges.adjacency,
                target_mask,
                plan,
            )
            return (
                self.configuration.alpha_bond * bond
                + self.configuration.alpha_adjacency * adjacency
            ) / scale

        return linear, quadratic

    def solve(
        self, predicted_graph: BatchedDenseData, targets: BatchedDenseData
    ) -> SolverResult:
        """Run the algorithm-specific optimization."""

        raise NotImplementedError

    def forward(
        self, predicted_graph: BatchedDenseData, targets: BatchedDenseData
    ) -> Tensor:
        """Return the detached plan from the algorithm-specific solve."""

        return self.solve(predicted_graph, targets).plan


class FrankWolfeSolver(QuadraticGraphMatchingSolver):
    """Solve graph matching with a CPU or GPU linear-assignment backend.

    The solver has no trainable state. Its output is a convex combination of
    permutation matrices, hence follows the unit row/column marginals required
    by the unchanged Any2GraphV2 reconstruction objective.
    """

    def __init__(
        self,
        objective_parameters: ObjectiveParameters | None = None,
        solver_parameters: SolverParameters | None = None,
    ) -> None:
        super().__init__(objective_parameters, solver_parameters)
        self.max_iterations = self.solver_configuration.max_iter_outer
        self.tolerance = self.solver_configuration.tol_outer
        self.backend = self.solver_configuration.solver_backend

    def _linear_assignment(self, cost: Tensor) -> Tensor:
        """Return batched permutation matrices with rows predicted, columns target."""

        if self.backend == "gpu":
            if not cost.is_cuda:
                raise RuntimeError("The GPU assignment backend requires a CUDA cost.")
            try:
                from torch_linear_assignment import _backend as tla_backend
                from torch_linear_assignment import batch_linear_assignment
            except ImportError as error:
                raise RuntimeError(
                    "solver_backend=gpu requires torch-linear-assignment."
                ) from error
            if not tla_backend.has_cuda():
                raise RuntimeError(
                    "torch-linear-assignment was installed without its CUDA "
                    "extension; reinstall it with FORCE_CUDA=1 and matching NVCC."
                )
            predicted_to_target = batch_linear_assignment(cost.contiguous())
            return torch.nn.functional.one_hot(
                predicted_to_target, num_classes=cost.shape[2]
            ).to(cost.dtype)

        matrices = []
        for batch_cost in cost.detach().cpu().numpy():
            predicted, target = linear_sum_assignment(batch_cost)
            matrix = torch.zeros(cost.shape[1:], dtype=cost.dtype)
            matrix[torch.from_numpy(predicted), torch.from_numpy(target)] = 1.0
            matrices.append(matrix)
        return torch.stack(matrices).to(cost.device)

    @staticmethod
    def _line_search(
        linear: Tensor,
        quadratic,
        plan: Tensor,
        vertex: Tensor,
    ) -> Tensor:
        delta = vertex - plan
        quadratic_delta = quadratic(delta)
        a = (delta * quadratic_delta).sum(dim=(1, 2))
        b = (linear * delta).sum(dim=(1, 2)) + 2.0 * (
            plan * quadratic_delta
        ).sum(dim=(1, 2))
        denominator = 2.0 * a.clamp_min(torch.finfo(a.dtype).eps)
        interior = (-b / denominator).clamp(0, 1)
        boundary = torch.where(a + b < 0, torch.ones_like(a), torch.zeros_like(a))
        step = torch.where(a > 0, interior, boundary)
        return plan + step[:, None, None] * delta

    def solve(
        self, predicted_graph: BatchedDenseData, targets: BatchedDenseData
    ) -> SolverResult:
        """Compute ``T[p,t]`` and retain per-graph convergence diagnostics."""

        self._validate(predicted_graph, targets)
        output_device = predicted_graph.h.device
        output_dtype = predicted_graph.h.dtype
        linear, quadratic = self._cost_terms(predicted_graph, targets)
        batch, size, _ = linear.shape
        plan = linear.new_full((batch, size, size), 1.0 / size)
        active = torch.ones(batch, device=linear.device, dtype=torch.bool)
        iterations = torch.zeros(batch, device=linear.device, dtype=torch.long)
        previous_cost = (plan * (linear + quadratic(plan))).sum(dim=(1, 2))

        for iteration in range(1, self.max_iterations + 1):
            gradient = linear + 2.0 * quadratic(plan)
            vertex = self._linear_assignment(gradient)
            updated = self._line_search(linear, quadratic, plan, vertex)
            cost = (updated * (linear + quadratic(updated))).sum(dim=(1, 2))
            converged = (previous_cost - cost).abs() < self.tolerance
            plan = torch.where(active[:, None, None], updated, plan)
            iterations = torch.where(active, iteration, iterations)
            active = active & ~converged
            previous_cost = torch.where(active, cost, previous_cost)
            if self.backend == "cpu" and not active.any():
                break

        return SolverResult(
            plan=plan.to(device=output_device, dtype=output_dtype),
            iterations=iterations.to(output_device),
        )

    def forward(
        self, predicted_graph: BatchedDenseData, targets: BatchedDenseData
    ) -> Tensor:
        """Return the detached plan directly for ``T = solver(prediction, target)``."""

        return self.solve(predicted_graph, targets).plan

    def hard_permutations(self, plan: Tensor) -> Tensor:
        """Project a soft solver plan to ``perm[target] = predicted`` indices."""

        matrix = self._linear_assignment(-plan)
        return matrix.argmax(dim=1)
