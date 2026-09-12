"""Project constants and command-line data parameters."""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence
from dataclasses import dataclass, fields
from pathlib import Path

import yaml


# Stable molecular vocabularies are code-level contracts, not experiment knobs.
VALID_ATOMS_LIST = (
    "C",
    "N",
    "O",
    "S",
    "P",
    "F",
    "Cl",
    "Br",
    "I",
    "B",
    "Si",
    "As",
    "Se",
    "Na",
    "K",
    "UNK",
)
VALID_BOND_TYPES = ("SINGLE", "DOUBLE", "TRIPLE", "AROMATIC")
COLORING_NODE_CLASSES = 4
COLORING_EDGE_CLASSES = 2
MORGAN_FINGERPRINT_RADIUS = 2
MORGAN_FINGERPRINT_SIZE = 2048
FINGERPRINT_UNK_TOKEN_ID = MORGAN_FINGERPRINT_SIZE
FINGERPRINT_SOS_TOKEN_ID = MORGAN_FINGERPRINT_SIZE + 1
FINGERPRINT_VOCAB_SIZE = MORGAN_FINGERPRINT_SIZE + 2


def _positive_integer(value: str) -> int:
    parsed = int(value)
    if parsed < 1:
        raise argparse.ArgumentTypeError("value must be a positive integer")
    return parsed


def _nonnegative_integer(value: str) -> int:
    parsed = int(value)
    if parsed < 0:
        raise argparse.ArgumentTypeError("value must be a non-negative integer")
    return parsed


def _positive_float(value: str) -> float:
    parsed = float(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("value must be positive")
    return parsed


def _nonnegative_float(value: str) -> float:
    parsed = float(value)
    if parsed < 0:
        raise argparse.ArgumentTypeError("value must be non-negative")
    return parsed


@dataclass(frozen=True)
class GraphParameters:
    """Experiment parameters controlling molecular graph construction."""

    n_nodes_max: int = 48
    scaffold: bool = False
    remove_invalid_molecules: bool = False

    def __post_init__(self) -> None:
        if self.n_nodes_max < 1:
            raise ValueError("n_nodes_max must be positive.")


@dataclass(frozen=True)
class DataParameters:
    """Shared data settings with one explicit task-routing parameter."""

    task: str = "ms2graph"
    data_dir: Path = Path("src/data/massspecgym")
    data_file: Path = Path("src/data/fingerprint2graph/4M_32.csv")
    split_method: str = "formula"
    train_iteration_mode: str = "molecule"
    eval_iteration_mode: str = "spectrum"
    spectrum_representation: str = "annotated"
    n_nodes_max: int = 32
    n_peaks_max: int = 128
    n_tokens_max: int = 256
    batch_size: int = 32
    num_workers: int = 0
    pin_memory: bool = False
    scaffold: bool = False
    remove_invalid_molecules: bool = False
    seed: int = 0

    def __post_init__(self) -> None:
        object.__setattr__(self, "data_dir", Path(self.data_dir))
        object.__setattr__(self, "data_file", Path(self.data_file))
        if self.task not in {"coloring2graph", "fingerprint2graph", "ms2graph"}:
            raise ValueError(
                "task must be coloring2graph, fingerprint2graph, or ms2graph."
            )
        if self.train_iteration_mode not in {"spectrum", "molecule"}:
            raise ValueError("train_iteration_mode must be spectrum or molecule.")
        if self.eval_iteration_mode not in {"spectrum", "molecule"}:
            raise ValueError("eval_iteration_mode must be spectrum or molecule.")
        if self.spectrum_representation not in {"annotated", "raw"}:
            raise ValueError("spectrum_representation must be annotated or raw.")
        if not self.split_method:
            raise ValueError("split_method cannot be empty.")
        if self.n_peaks_max < 1:
            raise ValueError("n_peaks_max must be positive.")
        if self.n_tokens_max < 2:
            raise ValueError("n_tokens_max must leave room for SOS and a bit token.")
        if self.batch_size < 1:
            raise ValueError("batch_size must be positive.")
        if self.num_workers < 0:
            raise ValueError("num_workers cannot be negative.")
        GraphParameters(
            n_nodes_max=self.n_nodes_max,
            scaffold=self.scaffold,
            remove_invalid_molecules=self.remove_invalid_molecules,
        )

    @property
    def graph(self) -> GraphParameters:
        return GraphParameters(
            n_nodes_max=self.n_nodes_max,
            scaffold=self.scaffold,
            remove_invalid_molecules=self.remove_invalid_molecules,
        )


@dataclass(frozen=True)
class ModelParameters:
    """Validated Transformer and graph-head hyperparameters."""

    graph_decoder_type: str = "node_query"
    d_token_input: int = 128
    d_node_decoder: int = 128
    n_heads: int = 8
    encoder_layers: int = 2
    decoder_layers: int = 2
    d_token_input_feedforward: int = 256
    d_node_decoder_feedforward: int = 256
    d_edge_decoder: int = 64
    dropout: float = 0.1
    use_collision_energy: bool = True
    collision_energy_max: float = 100.0
    use_token_positions: bool = True
    coloring_image_encoder_type: str = "resnet18"
    image_feature_grid_size: int = 16

    def __post_init__(self) -> None:
        if self.graph_decoder_type not in {"node_query", "relationformer"}:
            raise ValueError(
                "graph_decoder_type must be node_query or relationformer."
            )
        integer_fields = {
            "d_token_input": self.d_token_input,
            "d_node_decoder": self.d_node_decoder,
            "n_heads": self.n_heads,
            "encoder_layers": self.encoder_layers,
            "decoder_layers": self.decoder_layers,
            "d_token_input_feedforward": self.d_token_input_feedforward,
            "d_node_decoder_feedforward": self.d_node_decoder_feedforward,
            "d_edge_decoder": self.d_edge_decoder,
            "image_feature_grid_size": self.image_feature_grid_size,
        }
        for name, value in integer_fields.items():
            if value < 1:
                raise ValueError(f"{name} must be positive.")
        if self.d_token_input % self.n_heads != 0:
            raise ValueError("d_token_input must be divisible by n_heads.")
        if self.d_node_decoder % self.n_heads != 0:
            raise ValueError("d_node_decoder must be divisible by n_heads.")
        if self.coloring_image_encoder_type not in {"resnet18", "segformer_b1"}:
            raise ValueError(
                "coloring_image_encoder_type must be resnet18 or segformer_b1."
            )
        if not 0.0 <= self.dropout < 1.0:
            raise ValueError("dropout must be in [0, 1).")
        if (
            self.use_collision_energy or self.use_token_positions
        ) and self.d_token_input % 2:
            raise ValueError(
                "d_token_input must be even for enabled sinusoidal input encodings."
            )
        if self.collision_energy_max <= 0:
            raise ValueError("collision_energy_max must be positive.")


@dataclass(frozen=True)
class TargetEncoderParameters:
    """Validated selectable target-graph encoder hyperparameters."""

    target_encoder_type: str = "gnn"
    d_node_target: int = 128
    target_layers: int = 3
    target_dropout: float = 0.1
    use_laplacian_pe: bool = True
    laplacian_pe_dim: int = 8

    def __post_init__(self) -> None:
        if self.target_encoder_type not in {"gnn"}:
            raise ValueError(
                "target_encoder_type must be gnn."
            )
        if self.d_node_target < 1:
            raise ValueError("d_node_target must be positive.")
        if self.target_layers < 1:
            raise ValueError("target_layers must be positive.")
        if not 0.0 <= self.target_dropout < 1.0:
            raise ValueError("target_dropout must be in [0, 1).")
        if self.laplacian_pe_dim < 1:
            raise ValueError("laplacian_pe_dim must be positive.")


@dataclass(frozen=True)
class MatcherParameters:
    """Validated learned-matcher and transport hyperparameters."""

    matcher_type: str = "sinkhorn"
    matcher_dim: int = 128
    sinkhorn_mode: str = "unrolling"
    sinkhorn_iterations: int = 20
    sinkhorn_tolerance: float = 1e-6
    sinkhorn_check_convergence_every: int = 10
    matcher_epsilon: float = 1e-4
    normalize_matcher_cost: bool = True

    def __post_init__(self) -> None:
        if self.matcher_type not in {"sinkhorn", "softsort"}:
            raise ValueError("matcher_type must be sinkhorn or softsort.")
        if self.matcher_dim < 1:
            raise ValueError("matcher_dim must be positive.")
        if self.sinkhorn_mode not in {"unrolling", "implicit"}:
            raise ValueError("sinkhorn_mode must be unrolling or implicit.")
        if self.sinkhorn_iterations < 1:
            raise ValueError("sinkhorn_iterations must be positive.")
        if self.sinkhorn_tolerance <= 0:
            raise ValueError("sinkhorn_tolerance must be positive.")
        if self.sinkhorn_check_convergence_every < 1:
            raise ValueError(
                "sinkhorn_check_convergence_every must be positive."
            )
        if self.matcher_epsilon <= 0:
            raise ValueError("matcher_epsilon must be positive.")


@dataclass(frozen=True)
class SolverParameters:
    """Validated non-learned graph-matching solver parameters."""

    old_solver_approach: bool = False
    solver_type: str = "frank_wolfe"
    solver_backend: str = "cpu"
    tau: float = 0.1
    max_iter_inner: int = 100
    tol_inner: float = 1e-5
    max_iter_outer: int = 20
    tol_outer: float = 1e-5

    def __post_init__(self) -> None:
        if self.solver_type not in {"frank_wolfe", "mirror"}:
            raise ValueError("solver_type must be frank_wolfe or mirror.")
        if self.solver_backend not in {"cpu", "gpu"}:
            raise ValueError("solver_backend must be cpu or gpu.")
        if self.tau <= 0:
            raise ValueError("tau must be positive.")
        if self.max_iter_inner < 1:
            raise ValueError("max_iter_inner must be positive.")
        if self.tol_inner <= 0:
            raise ValueError("tol_inner must be positive.")
        if self.max_iter_outer < 1:
            raise ValueError("max_iter_outer must be positive.")
        if self.tol_outer <= 0:
            raise ValueError("tol_outer must be positive.")


@dataclass(frozen=True)
class ObjectiveParameters:
    """Validated molecular subset of the corrected GRALE objective."""

    reconstruction_loss: str = "original"
    alpha_presence: float = 1.0
    alpha_atom: float = 1.0
    alpha_bond: float = 0.2
    alpha_adjacency: float = 0.2
    alpha_marginal: float = 1.0
    feature_diffusion: bool = False
    alpha_feature_diffusion: float = 1.0
    exclude_self_loops: bool = False

    def __post_init__(self) -> None:
        if self.reconstruction_loss not in {
            "original",
            "alt_a",
            "alt_b",
            "alt_a_prime",
            "alt_b_prime",
        }:
            raise ValueError(
                "reconstruction_loss must be original, alt_a, alt_b, "
                "alt_a_prime, or alt_b_prime."
            )
        weights = {
            "alpha_presence": self.alpha_presence,
            "alpha_atom": self.alpha_atom,
            "alpha_bond": self.alpha_bond,
            "alpha_adjacency": self.alpha_adjacency,
            "alpha_marginal": self.alpha_marginal,
            "alpha_feature_diffusion": self.alpha_feature_diffusion,
        }
        for name, value in weights.items():
            if value < 0:
                raise ValueError(f"{name} must be non-negative.")
        active_weights = {
            name: value
            for name, value in weights.items()
            if name != "alpha_feature_diffusion" or self.feature_diffusion
        }
        if not any(value > 0 for value in active_weights.values()):
            raise ValueError("At least one objective weight must be positive.")


@dataclass(frozen=True)
class TrainingParameters:
    """Validated Lightning Trainer, optimizer, metric, and logger parameters."""

    learning_rate: float = 1e-4
    weight_decay: float = 0.0
    lr_scheduler: str = "constant"
    lr_warmup_fraction: float = 0.05
    min_learning_rate: float = 0.0
    max_epochs: int = 100
    gradient_clip_val: float = 1.0
    early_stopping_patience: int = 10
    early_stopping_min_delta: float = 0.0
    presence_threshold: float = 0.5
    accelerator: str = "auto"
    devices: int = 1
    precision: str = "32-true"
    deterministic: bool = False
    log_every_n_steps: int = 10
    checkpoint_every: int = 10_000
    limit_train_batches: int | None = None
    limit_val_batches: int | None = None
    validation_interval_minutes: int | None = None
    wandb_mode: str = "online"
    wandb_project: str = "any2graph-v2"
    wandb_entity: str | None = None
    run_name: str | None = None
    output_dir: Path = Path("artifacts")
    enable_progress_bar: bool = True
    checkpoint_path: Path | None = None
    enable_timing_metrics: bool = False
    timing_warmup_steps: int = 10
    timing_interval_steps: int = 10

    def __post_init__(self) -> None:
        object.__setattr__(self, "output_dir", Path(self.output_dir))
        if self.checkpoint_path is not None:
            object.__setattr__(self, "checkpoint_path", Path(self.checkpoint_path))
        if self.learning_rate <= 0:
            raise ValueError("learning_rate must be positive.")
        if self.weight_decay < 0:
            raise ValueError("weight_decay must be non-negative.")
        if self.lr_scheduler not in {"constant", "warmup_cosine"}:
            raise ValueError("lr_scheduler must be constant or warmup_cosine.")
        if not 0.0 <= self.lr_warmup_fraction < 1.0:
            raise ValueError("lr_warmup_fraction must be in [0, 1).")
        if self.min_learning_rate < 0:
            raise ValueError("min_learning_rate must be non-negative.")
        if self.min_learning_rate > self.learning_rate:
            raise ValueError("min_learning_rate cannot exceed learning_rate.")
        if self.max_epochs < 1:
            raise ValueError("max_epochs must be positive.")
        if self.gradient_clip_val < 0:
            raise ValueError("gradient_clip_val must be non-negative.")
        if self.early_stopping_patience < 0:
            raise ValueError("early_stopping_patience must be non-negative.")
        if self.early_stopping_min_delta < 0:
            raise ValueError("early_stopping_min_delta must be non-negative.")
        if not 0.0 <= self.presence_threshold <= 1.0:
            raise ValueError("presence_threshold must be in [0, 1].")
        if self.accelerator not in {"auto", "cpu", "gpu"}:
            raise ValueError("accelerator must be auto, cpu, or gpu.")
        if self.devices < 1:
            raise ValueError("devices must be positive.")
        if self.precision not in {"32-true", "16-mixed", "bf16-mixed"}:
            raise ValueError("Unsupported Lightning precision.")
        if self.checkpoint_every < 1:
            raise ValueError("checkpoint_every must be positive.")
        if self.log_every_n_steps < 1:
            raise ValueError("log_every_n_steps must be positive.")
        for name, value in {
            "limit_train_batches": self.limit_train_batches,
            "limit_val_batches": self.limit_val_batches,
            "validation_interval_minutes": self.validation_interval_minutes,
        }.items():
            if value is not None and value < 1:
                raise ValueError(f"{name} must be positive when provided.")
        if self.wandb_mode not in {"online", "offline", "disabled"}:
            raise ValueError("wandb_mode must be online, offline, or disabled.")
        if not self.wandb_project:
            raise ValueError("wandb_project cannot be empty.")
        if self.timing_warmup_steps < 0:
            raise ValueError("timing_warmup_steps cannot be negative.")
        if self.timing_interval_steps < 1:
            raise ValueError("timing_interval_steps must be positive.")


@dataclass(frozen=True)
class ExperimentParameters:
    """Data and predictor parameters parsed from one experiment command."""

    data: DataParameters
    model: ModelParameters


@dataclass(frozen=True)
class EncoderExperimentParameters:
    """Data, predictor, and target-encoder parameters for target encoding."""

    data: DataParameters
    model: ModelParameters
    target_encoder: TargetEncoderParameters


@dataclass(frozen=True)
class MatcherExperimentParameters:
    """Complete data, dual-encoder, and matcher configuration."""

    data: DataParameters
    model: ModelParameters
    target_encoder: TargetEncoderParameters
    matcher: MatcherParameters


@dataclass(frozen=True)
class ObjectiveExperimentParameters:
    """Complete pipeline and objective configuration."""

    data: DataParameters
    model: ModelParameters
    target_encoder: TargetEncoderParameters
    matcher: MatcherParameters
    solver: SolverParameters
    objective: ObjectiveParameters


@dataclass(frozen=True)
class TrainingExperimentParameters:
    """Complete data, model, objective, and Trainer configuration."""

    data: DataParameters
    model: ModelParameters
    target_encoder: TargetEncoderParameters
    matcher: MatcherParameters
    solver: SolverParameters
    objective: ObjectiveParameters
    training: TrainingParameters


def add_data_arguments(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    """Add Any2GraphV2 data arguments to a new or existing CLI parser."""

    defaults = DataParameters()
    group = parser.add_argument_group("data")
    group.add_argument(
        "--task",
        choices=("coloring2graph", "fingerprint2graph", "ms2graph"),
        default=defaults.task,
        help="select both the input dataloader and input encoder",
    )
    group.add_argument(
        "--data-dir",
        type=Path,
        default=defaults.data_dir,
        help="dataset directory (used by task=ms2graph or coloring2graph)",
    )
    group.add_argument(
        "--data-file",
        type=Path,
        default=defaults.data_file,
        help="preprocessed CSV (used by task=fingerprint2graph)",
    )
    group.add_argument(
        "--split-method",
        default=defaults.split_method,
        help="MassSpecGym split CSV basename",
    )
    group.add_argument(
        "--train-iteration-mode",
        choices=("spectrum", "molecule"),
        default=defaults.train_iteration_mode,
        help="MassSpecGym training sampling unit",
    )
    group.add_argument(
        "--eval-iteration-mode",
        choices=("spectrum", "molecule"),
        default=defaults.eval_iteration_mode,
        help="MassSpecGym evaluation sampling unit",
    )
    group.add_argument(
        "--spectrum-representation",
        choices=("annotated", "raw"),
        default=defaults.spectrum_representation,
        help="MassSpecGym spectrum input features",
    )
    group.add_argument(
        "--n-nodes-max",
        "--nodes-max",
        dest="n_nodes_max",
        type=_positive_integer,
        default=defaults.n_nodes_max,
        help="maximum number of target graph nodes",
    )
    group.add_argument(
        "--n-peaks-max",
        type=_positive_integer,
        default=defaults.n_peaks_max,
        help="maximum number of mass-spectrum peaks",
    )
    group.add_argument(
        "--n-tokens-max",
        type=_positive_integer,
        default=defaults.n_tokens_max,
        help="maximum SOS-plus-fingerprint-token sequence length",
    )
    group.add_argument(
        "--batch-size",
        type=_positive_integer,
        default=defaults.batch_size,
        help="DataLoader batch size",
    )
    group.add_argument(
        "--num-workers",
        type=_nonnegative_integer,
        default=defaults.num_workers,
        help="DataLoader worker count",
    )
    group.add_argument(
        "--pin-memory",
        action=argparse.BooleanOptionalAction,
        default=defaults.pin_memory,
        help="pin host tensors",
    )
    group.add_argument(
        "--scaffold",
        action=argparse.BooleanOptionalAction,
        default=defaults.scaffold,
        help="predict Murcko scaffolds instead of full molecules",
    )
    group.add_argument(
        "--remove-invalid-molecules",
        action=argparse.BooleanOptionalAction,
        default=defaults.remove_invalid_molecules,
        help="remove rather than truncate/fallback",
    )
    group.add_argument(
        "--seed", type=int, default=defaults.seed, help="sampling seed"
    )
    return parser


def add_model_arguments(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    """Add Any2GraphV2 predictor arguments to a new or existing CLI parser."""

    defaults = ModelParameters()
    group = parser.add_argument_group("model")
    group.add_argument(
        "--graph-decoder-type",
        choices=("node_query", "relationformer"),
        default=defaults.graph_decoder_type,
        help="standard node queries or Any2Graph-style Relationformer queries",
    )
    group.add_argument(
        "--d-token-input",
        type=_positive_integer,
        default=defaults.d_token_input,
        help="input-token Transformer width",
    )
    group.add_argument(
        "--d-node-decoder",
        type=_positive_integer,
        default=defaults.d_node_decoder,
        help="decoded node width and matcher interface width",
    )
    group.add_argument(
        "--n-heads", type=_positive_integer, default=defaults.n_heads,
        help="attention head count",
    )
    group.add_argument(
        "--encoder-layers", type=_positive_integer, default=defaults.encoder_layers,
        help="spectrum Transformer layer count",
    )
    group.add_argument(
        "--decoder-layers", type=_positive_integer, default=defaults.decoder_layers,
        help="node-query Transformer layer count",
    )
    group.add_argument(
        "--d-token-input-feedforward",
        type=_positive_integer,
        default=defaults.d_token_input_feedforward,
        help="input-token Transformer feed-forward width",
    )
    group.add_argument(
        "--d-node-decoder-feedforward",
        type=_positive_integer,
        default=defaults.d_node_decoder_feedforward,
        help="node decoder Transformer feed-forward width",
    )
    group.add_argument(
        "--d-edge-decoder",
        type=_positive_integer,
        default=defaults.d_edge_decoder,
        help="symmetric bond-head latent width",
    )
    group.add_argument(
        "--dropout", type=float, default=defaults.dropout,
        help="Transformer and head dropout",
    )
    group.add_argument(
        "--use-collision-energy",
        action=argparse.BooleanOptionalAction,
        default=defaults.use_collision_energy,
        help="condition the mass-spectrum encoder on collision energy",
    )
    group.add_argument(
        "--collision-energy-max",
        type=_positive_float,
        default=defaults.collision_energy_max,
        help="sinusoidal collision-energy scale",
    )
    group.add_argument(
        "--use-token-positions",
        action=argparse.BooleanOptionalAction,
        default=defaults.use_token_positions,
        help="add sinusoidal sequence positions to fingerprint token embeddings",
    )
    group.add_argument(
        "--coloring-image-encoder-type",
        choices=("resnet18", "segformer_b1"),
        default=defaults.coloring_image_encoder_type,
        help="Coloring RGB image backbone",
    )
    group.add_argument(
        "--image-feature-grid-size",
        type=_positive_integer,
        default=defaults.image_feature_grid_size,
        help="maximum Coloring encoder output-grid side",
    )
    return parser


def add_target_encoder_arguments(
    parser: argparse.ArgumentParser,
) -> argparse.ArgumentParser:
    """Add dense target-graph encoder options to a CLI parser."""

    defaults = TargetEncoderParameters()
    group = parser.add_argument_group("target graph encoder")
    group.add_argument(
        "--target-encoder-type",
        choices=("gnn",),
        default=defaults.target_encoder_type,
        help="target graph encoder architecture",
    )
    group.add_argument(
        "--d-node-target",
        type=_positive_integer,
        default=defaults.d_node_target,
        help="target GNN hidden width",
    )
    group.add_argument(
        "--target-layers",
        type=_positive_integer,
        default=defaults.target_layers,
        help="target message-passing layer count",
    )
    group.add_argument(
        "--target-dropout",
        type=float,
        default=defaults.target_dropout,
        help="target GNN dropout",
    )
    group.add_argument(
        "--use-laplacian-pe",
        action=argparse.BooleanOptionalAction,
        default=defaults.use_laplacian_pe,
        help="append Laplacian eigenvectors to target atom inputs",
    )
    group.add_argument(
        "--laplacian-pe-dim",
        type=_positive_integer,
        default=defaults.laplacian_pe_dim,
        help="maximum nonzero Laplacian eigenvectors per target",
    )
    return parser


def add_matcher_arguments(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    """Add learned matching and transport options to a CLI parser."""

    defaults = MatcherParameters()
    group = parser.add_argument_group("matcher")
    group.add_argument(
        "--matcher-type",
        choices=("sinkhorn", "softsort"),
        default=defaults.matcher_type,
        help="soft matching algorithm",
    )
    group.add_argument(
        "--matcher-dim",
        type=_positive_integer,
        default=defaults.matcher_dim,
        help="learned matching projection width",
    )
    group.add_argument(
        "--sinkhorn-iterations",
        type=_positive_integer,
        default=defaults.sinkhorn_iterations,
        help="fixed log-domain Sinkhorn update count",
    )
    group.add_argument(
        "--sinkhorn-mode",
        choices=("unrolling", "implicit"),
        default=defaults.sinkhorn_mode,
        help="differentiate through unrolled updates or use implicit differentiation",
    )
    group.add_argument(
        "--sinkhorn-tolerance",
        type=_positive_float,
        default=defaults.sinkhorn_tolerance,
        help="implicit Sinkhorn marginal-convergence tolerance",
    )
    group.add_argument(
        "--sinkhorn-check-convergence-every",
        type=_positive_integer,
        default=defaults.sinkhorn_check_convergence_every,
        help="implicit Sinkhorn convergence-check interval",
    )
    group.add_argument(
        "--matcher-epsilon",
        type=_positive_float,
        default=defaults.matcher_epsilon,
        help="matcher transport temperature",
    )
    group.add_argument(
        "--normalize-matcher-cost",
        action=argparse.BooleanOptionalAction,
        default=defaults.normalize_matcher_cost,
        help="normalize each pairwise cost matrix by its sum",
    )
    return parser


def add_solver_arguments(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    """Add non-learned graph-matching solver options."""

    defaults = SolverParameters()
    group = parser.add_argument_group("solver")
    group.add_argument(
        "--old-solver-approach",
        action=argparse.BooleanOptionalAction,
        default=defaults.old_solver_approach,
        help="replace the target encoder and learned matcher with a solver",
    )
    group.add_argument(
        "--solver-type",
        type=str.lower,
        choices=("frank_wolfe", "mirror"),
        default=defaults.solver_type,
        help="non-learned graph-matching algorithm",
    )
    group.add_argument(
        "--solver-backend",
        type=str.lower,
        choices=("cpu", "gpu"),
        default=defaults.solver_backend,
        help="Frank--Wolfe linear-assignment and compute backend",
    )
    group.add_argument(
        "--tau",
        type=_positive_float,
        default=defaults.tau,
        help="mirror-descent KL regularization weight",
    )
    group.add_argument(
        "--max-iter-inner",
        type=_positive_integer,
        default=defaults.max_iter_inner,
        help="maximum Sinkhorn iterations per mirror update",
    )
    group.add_argument(
        "--tol-inner",
        type=_positive_float,
        default=defaults.tol_inner,
        help="Sinkhorn marginal convergence tolerance",
    )
    group.add_argument(
        "--max-iter-outer",
        type=_positive_integer,
        default=defaults.max_iter_outer,
        help="maximum Frank--Wolfe or mirror outer iterations",
    )
    group.add_argument(
        "--tol-outer",
        type=_positive_float,
        default=defaults.tol_outer,
        help="outer objective-change or fixed-point convergence tolerance",
    )
    return parser


def add_objective_arguments(
    parser: argparse.ArgumentParser,
) -> argparse.ArgumentParser:
    """Add molecular reconstruction objective options to a CLI parser."""

    defaults = ObjectiveParameters()
    group = parser.add_argument_group("objective")
    group.add_argument(
        "--reconstruction-loss",
        choices=("original", "alt_a", "alt_b", "alt_a_prime", "alt_b_prime"),
        default=defaults.reconstruction_loss,
        help="graph reconstruction relaxation used for learned matching",
    )
    group.add_argument(
        "--alpha-presence",
        type=_nonnegative_float,
        default=defaults.alpha_presence,
        help="node-presence BCE weight",
    )
    group.add_argument(
        "--alpha-atom",
        type=_nonnegative_float,
        default=defaults.alpha_atom,
        help="categorical atom reconstruction weight",
    )
    group.add_argument(
        "--alpha-bond",
        type=_nonnegative_float,
        default=defaults.alpha_bond,
        help="categorical bond reconstruction weight",
    )
    group.add_argument(
        "--alpha-adjacency",
        type=_nonnegative_float,
        default=defaults.alpha_adjacency,
        help="binary adjacency reconstruction weight",
    )
    group.add_argument(
        "--alpha-marginal",
        type=_nonnegative_float,
        default=defaults.alpha_marginal,
        help="row/column MarginalKL weight",
    )
    group.add_argument(
        "--feature-diffusion",
        action=argparse.BooleanOptionalAction,
        default=defaults.feature_diffusion,
        help="predict and match the Any2Graph one-hop feature A@F",
    )
    group.add_argument(
        "--alpha-feature-diffusion",
        type=_nonnegative_float,
        default=defaults.alpha_feature_diffusion,
        help="squared-L2 A@F matching and reconstruction weight",
    )
    group.add_argument(
        "--exclude-self-loops",
        action=argparse.BooleanOptionalAction,
        default=defaults.exclude_self_loops,
        help="exclude predicted and target bond diagonals",
    )
    return parser


def add_training_arguments(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    """Add optimizer, Lightning Trainer, metric, and W&B options."""

    defaults = TrainingParameters()
    group = parser.add_argument_group("training")
    group.add_argument(
        "--learning-rate", type=_positive_float, default=defaults.learning_rate
    )
    group.add_argument(
        "--weight-decay", type=_nonnegative_float, default=defaults.weight_decay
    )
    group.add_argument(
        "--lr-scheduler",
        choices=("constant", "warmup_cosine"),
        default=defaults.lr_scheduler,
        help="constant learning rate or step-wise linear-warmup cosine decay",
    )
    group.add_argument(
        "--lr-warmup-fraction",
        type=float,
        default=defaults.lr_warmup_fraction,
        help="fraction of estimated optimizer steps used for linear warmup",
    )
    group.add_argument(
        "--min-learning-rate",
        type=_nonnegative_float,
        default=defaults.min_learning_rate,
        help="terminal learning rate for warmup_cosine scheduling",
    )
    group.add_argument(
        "--max-epochs", type=_positive_integer, default=defaults.max_epochs
    )
    group.add_argument(
        "--gradient-clip-val",
        type=_nonnegative_float,
        default=defaults.gradient_clip_val,
    )
    group.add_argument(
        "--early-stopping-patience",
        type=_nonnegative_integer,
        default=defaults.early_stopping_patience,
    )
    group.add_argument(
        "--early-stopping-min-delta",
        type=_nonnegative_float,
        default=defaults.early_stopping_min_delta,
    )
    group.add_argument(
        "--presence-threshold", type=float, default=defaults.presence_threshold
    )
    group.add_argument(
        "--accelerator",
        choices=("auto", "cpu", "gpu"),
        default=defaults.accelerator,
    )
    group.add_argument("--devices", type=_positive_integer, default=defaults.devices)
    group.add_argument(
        "--precision",
        choices=("32-true", "16-mixed", "bf16-mixed"),
        default=defaults.precision,
    )
    group.add_argument(
        "--deterministic",
        action=argparse.BooleanOptionalAction,
        default=defaults.deterministic,
    )
    group.add_argument(
        "--log-every-n-steps",
        type=_positive_integer,
        default=defaults.log_every_n_steps,
    )
    group.add_argument(
        "--checkpoint-every",
        type=_positive_integer,
        default=defaults.checkpoint_every,
        help="save a full checkpoint every N optimizer steps",
    )
    group.add_argument(
        "--limit-train-batches",
        type=_positive_integer,
        default=defaults.limit_train_batches,
        help="optional integer batch limit per training epoch",
    )
    group.add_argument(
        "--limit-val-batches",
        type=_positive_integer,
        default=defaults.limit_val_batches,
        help="optional integer batch limit per validation epoch",
    )
    group.add_argument(
        "--validation-interval-minutes",
        type=_positive_integer,
        default=defaults.validation_interval_minutes,
        help=(
            "optional wall-clock interval for additional full validation passes; "
            "epoch-end validation remains enabled"
        ),
    )
    group.add_argument(
        "--wandb-mode",
        choices=("online", "offline", "disabled"),
        default=defaults.wandb_mode,
    )
    group.add_argument("--wandb-project", default=defaults.wandb_project)
    group.add_argument("--wandb-entity", default=defaults.wandb_entity)
    group.add_argument("--run-name", default=defaults.run_name)
    group.add_argument("--output-dir", type=Path, default=defaults.output_dir)
    group.add_argument(
        "--enable-progress-bar",
        action=argparse.BooleanOptionalAction,
        default=defaults.enable_progress_bar,
    )
    group.add_argument(
        "--checkpoint-path",
        type=Path,
        default=defaults.checkpoint_path,
        help="optional Lightning checkpoint from which to resume training",
    )
    group.add_argument(
        "--enable-timing-metrics",
        action=argparse.BooleanOptionalAction,
        default=defaults.enable_timing_metrics,
        help="sample synchronized CUDA transport-plan and training-step timings",
    )
    group.add_argument(
        "--timing-warmup-steps",
        type=_nonnegative_integer,
        default=defaults.timing_warmup_steps,
        help="optimizer steps excluded before timing begins",
    )
    group.add_argument(
        "--timing-interval-steps",
        type=_positive_integer,
        default=defaults.timing_interval_steps,
        help="optimizer-step interval between synchronized timing samples",
    )
    return parser


def data_argument_parser() -> argparse.ArgumentParser:
    """Create the standalone parser used by data-loading commands."""

    parser = argparse.ArgumentParser(
        description="Load Any2GraphV2 data",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    return add_data_arguments(parser)


def parse_data_parameters(arguments: Sequence[str] | None = None) -> DataParameters:
    """Parse command-line arguments into the internal parameter object."""

    namespace = data_argument_parser().parse_args(arguments)
    return DataParameters(**vars(namespace))


def model_argument_parser() -> argparse.ArgumentParser:
    """Create the standalone parser for spectrum predictor parameters."""

    parser = argparse.ArgumentParser(
        description="Configure the Any2GraphV2 spectrum predictor",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    return add_model_arguments(parser)


def parse_model_parameters(arguments: Sequence[str] | None = None) -> ModelParameters:
    """Parse model command-line arguments into a validated parameter object."""

    namespace = model_argument_parser().parse_args(arguments)
    try:
        return ModelParameters(**vars(namespace))
    except ValueError as error:
        model_argument_parser().error(str(error))


def experiment_argument_parser() -> argparse.ArgumentParser:
    """Create a parser containing both data and spectrum-predictor options."""

    parser = argparse.ArgumentParser(
        description="Run the Any2GraphV2 spectrum predictor",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    add_data_arguments(parser)
    add_model_arguments(parser)
    return parser


def parse_experiment_parameters(
    arguments: Sequence[str] | None = None,
) -> ExperimentParameters:
    """Parse a complete data-plus-model experiment configuration."""

    parser = experiment_argument_parser()
    values = vars(parser.parse_args(arguments))
    data_names = {field.name for field in fields(DataParameters)}
    model_names = {field.name for field in fields(ModelParameters)}
    try:
        return ExperimentParameters(
            data=DataParameters(**{name: values[name] for name in data_names}),
            model=ModelParameters(**{name: values[name] for name in model_names}),
        )
    except ValueError as error:
        parser.error(str(error))


def encoder_experiment_argument_parser() -> argparse.ArgumentParser:
    """Create the complete data and dual-encoder parser."""

    parser = argparse.ArgumentParser(
        description="Encode predicted and target Any2GraphV2 nodes",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    add_data_arguments(parser)
    add_model_arguments(parser)
    add_target_encoder_arguments(parser)
    return parser


def parse_encoder_experiment_parameters(
    arguments: Sequence[str] | None = None,
) -> EncoderExperimentParameters:
    """Parse data, predictor, and target-encoder arguments together."""

    parser = encoder_experiment_argument_parser()
    values = vars(parser.parse_args(arguments))
    data_names = {field.name for field in fields(DataParameters)}
    model_names = {field.name for field in fields(ModelParameters)}
    target_names = {field.name for field in fields(TargetEncoderParameters)}
    try:
        return EncoderExperimentParameters(
            data=DataParameters(**{name: values[name] for name in data_names}),
            model=ModelParameters(**{name: values[name] for name in model_names}),
            target_encoder=TargetEncoderParameters(
                **{name: values[name] for name in target_names}
            ),
        )
    except ValueError as error:
        parser.error(str(error))


def matcher_experiment_argument_parser() -> argparse.ArgumentParser:
    """Create the complete parser."""

    parser = argparse.ArgumentParser(
        description="Match Any2GraphV2 predicted and target nodes",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    add_data_arguments(parser)
    add_model_arguments(parser)
    add_target_encoder_arguments(parser)
    add_matcher_arguments(parser)
    return parser


def parse_matcher_experiment_parameters(
    arguments: Sequence[str] | None = None,
) -> MatcherExperimentParameters:
    """Parse the complete dual-encoder and matcher configuration."""

    parser = matcher_experiment_argument_parser()
    values = vars(parser.parse_args(arguments))
    data_names = {field.name for field in fields(DataParameters)}
    model_names = {field.name for field in fields(ModelParameters)}
    target_names = {field.name for field in fields(TargetEncoderParameters)}
    matcher_names = {field.name for field in fields(MatcherParameters)}
    try:
        return MatcherExperimentParameters(
            data=DataParameters(**{name: values[name] for name in data_names}),
            model=ModelParameters(**{name: values[name] for name in model_names}),
            target_encoder=TargetEncoderParameters(
                **{name: values[name] for name in target_names}
            ),
            matcher=MatcherParameters(
                **{name: values[name] for name in matcher_names}
            ),
        )
    except ValueError as error:
        parser.error(str(error))


def objective_experiment_argument_parser() -> argparse.ArgumentParser:
    """Create the complete parser."""

    parser = argparse.ArgumentParser(
        description="Evaluate the Any2GraphV2 reconstruction objective",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    add_data_arguments(parser)
    add_model_arguments(parser)
    add_target_encoder_arguments(parser)
    add_matcher_arguments(parser)
    add_solver_arguments(parser)
    add_objective_arguments(parser)
    return parser


def parse_objective_experiment_parameters(
    arguments: Sequence[str] | None = None,
) -> ObjectiveExperimentParameters:
    """Parse the complete model, matcher, and objective configuration."""

    parser = objective_experiment_argument_parser()
    values = vars(parser.parse_args(arguments))
    data_names = {field.name for field in fields(DataParameters)}
    model_names = {field.name for field in fields(ModelParameters)}
    target_names = {field.name for field in fields(TargetEncoderParameters)}
    matcher_names = {field.name for field in fields(MatcherParameters)}
    solver_names = {field.name for field in fields(SolverParameters)}
    objective_names = {field.name for field in fields(ObjectiveParameters)}
    try:
        return ObjectiveExperimentParameters(
            data=DataParameters(**{name: values[name] for name in data_names}),
            model=ModelParameters(**{name: values[name] for name in model_names}),
            target_encoder=TargetEncoderParameters(
                **{name: values[name] for name in target_names}
            ),
            matcher=MatcherParameters(
                **{name: values[name] for name in matcher_names}
            ),
            solver=SolverParameters(
                **{name: values[name] for name in solver_names}
            ),
            objective=ObjectiveParameters(
                **{name: values[name] for name in objective_names}
            ),
        )
    except ValueError as error:
        parser.error(str(error))


def training_experiment_argument_parser() -> argparse.ArgumentParser:
    """Create the complete training parser."""

    parser = argparse.ArgumentParser(
        description="Train Any2GraphV2 with Lightning and W&B",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help=(
            "YAML configuration path or short name resolved as configs/<name>.yaml; "
            "later CLI options override YAML values"
        ),
    )
    add_data_arguments(parser)
    add_model_arguments(parser)
    add_target_encoder_arguments(parser)
    add_matcher_arguments(parser)
    add_solver_arguments(parser)
    add_objective_arguments(parser)
    add_training_arguments(parser)
    return parser


_CONFIG_PARAMETER_TYPES = {
    "data": DataParameters,
    "model": ModelParameters,
    "target_encoder": TargetEncoderParameters,
    "matcher": MatcherParameters,
    "solver": SolverParameters,
    "objective": ObjectiveParameters,
    "training": TrainingParameters,
}
_NULLABLE_CONFIG_FIELDS = {
    "limit_train_batches",
    "limit_val_batches",
    "validation_interval_minutes",
    "wandb_entity",
    "run_name",
    "checkpoint_path",
}


def resolve_training_config_path(config: str | Path) -> Path:
    """Resolve a YAML config path, accepting bare names such as ``default``."""

    requested = Path(config)
    candidates = [requested]
    if requested.suffix not in {".yaml", ".yml"}:
        candidates.append(requested.with_suffix(".yaml"))
    if not requested.is_absolute() and requested.parent == Path("."):
        candidates.append(Path("configs") / requested.with_suffix(".yaml"))
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    # Report the conventional location for short names in the argparse error.
    return candidates[-1]


def _config_to_arguments(
    config_path: Path, parser: argparse.ArgumentParser
) -> list[str]:
    """Validate a human-themed YAML configuration and create argparse tokens."""

    try:
        document = yaml.safe_load(config_path.read_text())
    except OSError as error:
        parser.error(f"cannot read config file {config_path}: {error}")
    except yaml.YAMLError as error:
        parser.error(f"invalid YAML in config file {config_path}: {error}")
    if not isinstance(document, dict):
        parser.error(f"config file {config_path} must contain a YAML mapping")

    valid_names = {
        field.name
        for parameter_type in _CONFIG_PARAMETER_TYPES.values()
        for field in fields(parameter_type)
    }
    arguments: list[str] = []
    seen: set[str] = set()
    for section, section_values in document.items():
        if not isinstance(section_values, dict):
            parser.error(f"config section {section!r} must contain a YAML mapping")
        for name, value in section_values.items():
            if name not in valid_names:
                parser.error(f"unknown parameter {section}.{name} in {config_path}")
            if name in seen:
                parser.error(f"duplicate parameter {name!r} in {config_path}")
            seen.add(name)
            option = f"--{name.replace('_', '-')}"
            if value is None:
                if name not in _NULLABLE_CONFIG_FIELDS:
                    parser.error(f"parameter {section}.{name} cannot be null")
                continue
            if isinstance(value, bool):
                arguments.append(option if value else f"--no-{option[2:]}")
            elif isinstance(value, (str, int, float)):
                arguments.extend((option, str(value)))
            else:
                parser.error(
                    f"parameter {section}.{name} must be a scalar YAML value"
                )
    return arguments


def parse_training_experiment_parameters(
    arguments: Sequence[str] | None = None,
) -> TrainingExperimentParameters:
    """Parse the complete experiment configuration."""

    parser = training_experiment_argument_parser()
    raw_arguments = list(arguments) if arguments is not None else sys.argv[1:]
    config_parser = argparse.ArgumentParser(add_help=False)
    config_parser.add_argument("--config", type=str)
    config_values, remaining = config_parser.parse_known_args(raw_arguments)
    if config_values.config is not None:
        config_path = resolve_training_config_path(config_values.config)
        config_arguments = _config_to_arguments(config_path, parser)
        raw_arguments = config_arguments + remaining
    values = vars(parser.parse_args(raw_arguments))
    parameter_types = (
        DataParameters,
        ModelParameters,
        TargetEncoderParameters,
        MatcherParameters,
        SolverParameters,
        ObjectiveParameters,
        TrainingParameters,
    )
    names = {
        parameter_type: {field.name for field in fields(parameter_type)}
        for parameter_type in parameter_types
    }
    try:
        return TrainingExperimentParameters(
            data=DataParameters(**{name: values[name] for name in names[DataParameters]}),
            model=ModelParameters(
                **{name: values[name] for name in names[ModelParameters]}
            ),
            target_encoder=TargetEncoderParameters(
                **{name: values[name] for name in names[TargetEncoderParameters]}
            ),
            matcher=MatcherParameters(
                **{name: values[name] for name in names[MatcherParameters]}
            ),
            solver=SolverParameters(
                **{name: values[name] for name in names[SolverParameters]}
            ),
            objective=ObjectiveParameters(
                **{name: values[name] for name in names[ObjectiveParameters]}
            ),
            training=TrainingParameters(
                **{name: values[name] for name in names[TrainingParameters]}
            ),
        )
    except ValueError as error:
        parser.error(str(error))


if __name__ == "__main__":
    print(parse_data_parameters())
