"""Non-learned graph-matching solver interfaces and implementations."""

from any2graph_v2.parameter import ObjectiveParameters, SolverParameters

from .frank_wolfe import (
    FrankWolfeSolver,
    SolverResult,
)
from .mirror import MirrorSolver


def build_solver(
    objective_parameters: ObjectiveParameters | None = None,
    solver_parameters: SolverParameters | None = None,
) -> FrankWolfeSolver | MirrorSolver:
    """Construct the selected non-learned graph-matching solver."""

    configuration = solver_parameters or SolverParameters()
    objective_configuration = objective_parameters or ObjectiveParameters()
    if objective_configuration.reconstruction_loss != "original":
        raise ValueError(
            "Non-learned solvers optimize only reconstruction_loss=original."
        )
    solver_class = {
        "frank_wolfe": FrankWolfeSolver,
        "mirror": MirrorSolver,
    }[configuration.solver_type]
    return solver_class(objective_configuration, configuration)


__all__ = [
    "FrankWolfeSolver",
    "MirrorSolver",
    "SolverResult",
    "build_solver",
]
