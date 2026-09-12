"""Batched CUDA mirror descent for quadratic graph matching."""

from __future__ import annotations

import numpy as np
import torch
from scipy.optimize import linear_sum_assignment
from torch import Tensor

from any2graph_v2.data.dense_data import BatchedDenseData
from any2graph_v2.models.matcher import sinkhorn_tol
from any2graph_v2.parameter import ObjectiveParameters, SolverParameters

from .frank_wolfe import QuadraticGraphMatchingSolver, SolverResult


def sinkhorn_mirror_step(
    gradient: Tensor,
    plan: Tensor,
    *,
    tau: float,
    max_iter_inner: int,
    tol_inner: float,
) -> Tensor:
    """Solve one KL-proximal linearized subproblem with batched Sinkhorn.

    The effective entropic-transport cost is
    ``gradient - tau * log(plan)``. Passing that cost with entropy weight
    ``tau`` to Sinkhorn produces a kernel proportional to
    ``plan * exp(-gradient / tau)``.
    """

    if gradient.shape != plan.shape or gradient.ndim != 3:
        raise ValueError(
            "gradient and plan must have shape [batch, predicted, target]."
        )
    if tau <= 0 or max_iter_inner < 1 or tol_inner <= 0:
        raise ValueError("Mirror and inner Sinkhorn parameters must be positive.")
    floor = torch.finfo(plan.dtype).tiny
    effective_cost = gradient - tau * plan.clamp_min(floor).log()
    return sinkhorn_tol(
        effective_cost,
        epsilon=tau,
        max_iter=max_iter_inner,
        tol=tol_inner,
        check_convergence_every=1,
    )


class MirrorSolver(QuadraticGraphMatchingSolver):
    """Solve graph matching by KL-proximal mirror descent entirely on CUDA.

    Cost construction is inherited from the Frank--Wolfe implementation so
    both solvers optimize the same Any2GraphV2 quadratic objective. The outer
    updates and every inner Sinkhorn problem are batched Torch CUDA operations.
    """

    def __init__(
        self,
        objective_parameters: ObjectiveParameters | None = None,
        solver_parameters: SolverParameters | None = None,
    ) -> None:
        configuration = solver_parameters or SolverParameters(solver_type="mirror")
        super().__init__(objective_parameters, configuration)
        self.backend = "gpu"
        self.tau = configuration.tau
        self.max_iter_inner = configuration.max_iter_inner
        self.tol_inner = configuration.tol_inner
        self.max_iter_outer = configuration.max_iter_outer
        self.tol_outer = configuration.tol_outer

    def solve(
        self, predicted_graph: BatchedDenseData, targets: BatchedDenseData
    ) -> SolverResult:
        """Return the detached, approximately bistochastic mirror fixed point."""

        self._validate(predicted_graph, targets)
        if not predicted_graph.h.is_cuda:
            raise RuntimeError("MirrorSolver requires predictions and targets on CUDA.")
        output_dtype = predicted_graph.h.dtype
        linear, quadratic = self._cost_terms(predicted_graph, targets)
        batch, size, _ = linear.shape
        plan = linear.new_full((batch, size, size), 1.0 / size)
        active = torch.ones(batch, device=linear.device, dtype=torch.bool)
        iterations = torch.zeros(batch, device=linear.device, dtype=torch.long)

        for iteration in range(1, self.max_iter_outer + 1):
            gradient = linear + 2.0 * quadratic(plan)
            updated = sinkhorn_mirror_step(
                gradient,
                plan,
                tau=self.tau,
                max_iter_inner=self.max_iter_inner,
                tol_inner=self.tol_inner,
            )
            residual = (updated - plan).abs().amax(dim=(1, 2))
            converged = residual < self.tol_outer
            plan = torch.where(active[:, None, None], updated, plan)
            iterations = torch.where(active, iteration, iterations)
            active = active & ~converged

        return SolverResult(plan=plan.to(output_dtype), iterations=iterations)

    def hard_permutations(self, plan: Tensor) -> Tensor:
        """Project plans for evaluation; this is outside the mirror optimization."""

        permutations = []
        for batch_plan in plan.detach().cpu().numpy():
            _, predicted_for_target = linear_sum_assignment(-batch_plan.T)
            permutations.append(predicted_for_target)
        return torch.from_numpy(np.stack(permutations)).to(
            device=plan.device, dtype=torch.long
        )
