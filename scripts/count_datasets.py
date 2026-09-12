"""Print strict dataset sizes for the two primary molecular prediction tasks.

Run from the repository root with ``PYTHONPATH=src python scripts/count_datasets.py``.
Fingerprint CSVs without the
persistent graph metadata introduced by preprocessing are upgraded once by the
normal datamodule path; subsequent counts only scan their stored metadata.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from any2graph_v2.data import FingerprintGraphDataModule, MassSpecGymDataModule
from any2graph_v2.parameter import DataParameters


PROJECT_ROOT = Path(__file__).resolve().parents[1]
N_NODES_MAX = 32
FOLDS = ("train", "val", "test")


def _parser() -> argparse.ArgumentParser:
    """Build the small development-script argument parser."""

    parser = argparse.ArgumentParser(
        description=(
            "Print strict MS2Scaffold and Fingerprint2Mol dataset sizes for "
            f"graphs with at most {N_NODES_MAX} nodes."
        )
    )
    parser.add_argument(
        "--massspec-dir",
        type=Path,
        default=PROJECT_ROOT / "src/data/massspecgym",
        help="MassSpecGym dataset directory",
    )
    parser.add_argument(
        "--fingerprint-file",
        type=Path,
        default=PROJECT_ROOT / "src/data/fingerprint2graph/4M_32.csv",
        help="preprocessed Fingerprint2Mol CSV (for example PUBCHEM_32.csv)",
    )
    parser.add_argument(
        "--split-method",
        default="formula",
        help="MassSpecGym split filename without the .csv suffix",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=8,
        help="workers used only for a one-time fingerprint metadata upgrade",
    )
    return parser


def _print_ms2scaffold_sizes(
    data_dir: Path, split_method: str
) -> None:
    """Build the strict scaffold index and print spectrum/molecule counts."""

    module = MassSpecGymDataModule(
        DataParameters(
            task="ms2graph",
            data_dir=data_dir,
            split_method=split_method,
            spectrum_representation="raw",
            train_iteration_mode="spectrum",
            eval_iteration_mode="spectrum",
            n_nodes_max=N_NODES_MAX,
            scaffold=True,
            remove_invalid_molecules=True,
        )
    )
    module.setup("fit")
    datasets = {
        "train": module.train_dataset,
        "val": module.val_dataset,
        "test": module.test_dataset,
    }
    if any(dataset is None for dataset in datasets.values()):
        raise RuntimeError("MS2Scaffold datamodule setup did not create every fold.")

    print(f"\nMS2Scaffold (n_nodes_max={N_NODES_MAX}, split={split_method})")
    total_spectra = 0
    total_molecules = 0
    for fold in FOLDS:
        dataset = datasets[fold]
        assert dataset is not None
        spectra = len(dataset.spectrum_indices)
        molecules = len(dataset.molecule_indices)
        total_spectra += spectra
        total_molecules += molecules
        print(f"  {fold:>5}: {spectra:>9,} spectra; {molecules:>8,} molecules")
    print(
        f"  total: {total_spectra:>9,} spectra; "
        f"{total_molecules:>8,} fold-specific molecules"
    )


def _print_fingerprint2mol_sizes(data_file: Path, workers: int) -> None:
    """Build the strict molecular CSV index and print example counts."""

    module = FingerprintGraphDataModule(
        DataParameters(
            task="fingerprint2graph",
            data_file=data_file,
            n_nodes_max=N_NODES_MAX,
            scaffold=False,
            remove_invalid_molecules=True,
            num_workers=workers,
        )
    )
    module.setup("fit")
    datasets = {
        "train": module.train_dataset,
        "val": module.val_dataset,
        "test": module.test_dataset,
    }
    if any(dataset is None for dataset in datasets.values()):
        raise RuntimeError("Fingerprint2Mol datamodule setup did not create every fold.")

    print(f"\nFingerprint2Mol (n_nodes_max={N_NODES_MAX}, file={data_file})")
    total = 0
    for fold in FOLDS:
        dataset = datasets[fold]
        assert dataset is not None
        size = len(dataset)
        total += size
        print(f"  {fold:>5}: {size:>9,} molecules")
    print(f"  total: {total:>9,} molecules")


def main() -> None:
    """Print strict per-fold sizes for MS2Scaffold and Fingerprint2Mol."""

    args = _parser().parse_args()
    if args.workers < 1:
        raise ValueError("--workers must be at least 1.")
    _print_ms2scaffold_sizes(args.massspec_dir, args.split_method)
    _print_fingerprint2mol_sizes(args.fingerprint_file, args.workers)


if __name__ == "__main__":
    main()
