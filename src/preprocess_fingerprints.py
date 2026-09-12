"""Stream raw SMILES files into filtered Fingerprint2Graph CSV datasets."""

from __future__ import annotations

import argparse
import csv
import gzip
import os
import shutil
import tempfile
import time
from contextlib import closing
from collections import Counter
from collections import deque
from collections.abc import Iterator, Sequence
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from itertools import islice
from pathlib import Path
from typing import Any
from urllib.request import urlopen

import numpy as np
from rdkit import Chem, RDLogger, rdBase
from rdkit.Chem.Scaffolds import MurckoScaffold
from tqdm.auto import tqdm

from any2graph_v2.parameter import VALID_ATOMS_LIST, VALID_BOND_TYPES


_SUPPORTED_BOND_TYPES = frozenset(VALID_BOND_TYPES)
_SUPPORTED_ATOMS = frozenset(VALID_ATOMS_LIST) - {"UNK"}
FINGERPRINT_GRAPH_METADATA_COLUMNS = (
    "molecule_size",
    "molecule_scaffold_size",
    "graph_valid",
)
PUBCHEM_CID_SMILES_URL = (
    "https://ftp.ncbi.nlm.nih.gov/pubchem/Compound/Extras/CID-SMILES.gz"
)
DEFAULT_PUBCHEM_RAW_PATH = Path("src/data/fingerprint2graph/PUBCHEM_raw.csv")


def ensure_pubchem_raw_dataset(
    path: Path = DEFAULT_PUBCHEM_RAW_PATH,
    *,
    url: str = PUBCHEM_CID_SMILES_URL,
) -> Path:
    """Download and atomically unpack PubChem CID--SMILES data when absent.

    PubChem distributes this source as a gzip-compressed, headerless TSV.  The
    project keeps the unpacked stream at ``path`` because preprocessing can
    then iterate it without a decompression step on every invocation.
    """

    if path.is_file():
        print(f"[data] PubChem raw source already exists: {path}", flush=True)
        return path
    if path.exists():
        raise ValueError(f"PubChem raw source path is not a file: {path}")

    path.parent.mkdir(parents=True, exist_ok=True)
    print(f"[data] PubChem raw source missing: {path}", flush=True)
    print(f"[data] downloading PubChem CID--SMILES from {url}...", flush=True)
    started_at = time.monotonic()
    compressed_descriptor, compressed_name = tempfile.mkstemp(
        dir=path.parent, prefix=f".{path.name}.", suffix=".gz.download"
    )
    raw_descriptor, raw_name = tempfile.mkstemp(
        dir=path.parent, prefix=f".{path.name}.", suffix=".unpack"
    )
    os.close(raw_descriptor)
    compressed_path = Path(compressed_name)
    raw_path = Path(raw_name)
    try:
        with os.fdopen(compressed_descriptor, "wb") as destination:
            with closing(urlopen(url, timeout=60)) as response:  # nosec B310: fixed HTTPS URL
                shutil.copyfileobj(response, destination, length=1024 * 1024)
        print("[data] unpacking PubChem CID--SMILES into the raw dataset...", flush=True)
        with (
            gzip.open(compressed_path, "rb") as source,
            raw_path.open("wb") as destination,
        ):
            shutil.copyfileobj(source, destination, length=1024 * 1024)
        os.replace(raw_path, path)
    except BaseException:
        raw_path.unlink(missing_ok=True)
        raise
    finally:
        compressed_path.unlink(missing_ok=True)
    elapsed = time.monotonic() - started_at
    print(f"[data] PubChem raw source ready: {path} ({elapsed:.1f}s)", flush=True)
    return path


@dataclass(frozen=True)
class MoleculeMetadata:
    """Reusable strict-graph eligibility facts derived from one SMILES string.

    ``molecule_size`` is the canonical full-molecule heavy-atom count.
    ``molecule_scaffold_size`` is zero for a valid ringless molecule and minus
    one only when RDKit cannot construct its scaffold. ``graph_valid`` covers
    the project's strict atom and bond vocabularies, independently of a chosen
    graph-size cap.
    """

    molecule_size: int
    molecule_scaffold_size: int
    graph_valid: bool

    def as_csv_row(self) -> dict[str, str]:
        """Return stable string values suitable for ``csv.DictWriter``."""

        return {
            "molecule_size": str(self.molecule_size),
            "molecule_scaffold_size": str(self.molecule_scaffold_size),
            "graph_valid": "true" if self.graph_valid else "false",
        }


def _disable_rdkit_diagnostics() -> None:
    """Initialize each worker without noisy per-molecule parse diagnostics."""

    RDLogger.DisableLog("rdApp.error")
    RDLogger.DisableLog("rdApp.warning")


def molecule_metadata(smiles: str) -> tuple[str, MoleculeMetadata]:
    """Classify one SMILES and compute metadata used by strict graph loading.

    The status controls preprocessing retention: malformed, empty, and
    unsupported-bond molecules are discarded. Molecules with unsupported atom
    symbols are retained for non-strict ``UNK`` handling but have
    ``graph_valid=false`` and are excluded by strict loading.
    """

    empty = MoleculeMetadata(0, -1, False)
    try:
        with rdBase.BlockLogs():
            parsed = Chem.MolFromSmiles(smiles)
            if parsed is None:
                return "invalid_smiles", empty
            molecule = Chem.RemoveHs(parsed)
            canonical_smiles = Chem.MolToSmiles(molecule, canonical=True)
            molecule = Chem.MolFromSmiles(canonical_smiles)
            if molecule is None:
                return "invalid_smiles", empty
            molecule_size = molecule.GetNumHeavyAtoms()
            if molecule_size == 0:
                return "empty_molecule", empty
            if any(
                str(bond.GetBondType()) not in _SUPPORTED_BOND_TYPES
                for bond in molecule.GetBonds()
            ):
                return "unsupported_bond", MoleculeMetadata(
                    molecule_size, -1, False
                )
            try:
                scaffold_size = MurckoScaffold.GetScaffoldForMol(
                    molecule
                ).GetNumHeavyAtoms()
            except Exception:
                scaffold_size = -1
            graph_valid = all(
                atom.GetSymbol() in _SUPPORTED_ATOMS for atom in molecule.GetAtoms()
            )
            return "kept", MoleculeMetadata(
                molecule_size=molecule_size,
                molecule_scaffold_size=scaffold_size,
                graph_valid=graph_valid,
            )
    except Exception:
        return "rdkit_error", empty


def _classify_batch(
    rows: list[dict[str, str]], atom_limit: int
) -> list[tuple[dict[str, str], str, MoleculeMetadata]]:
    """Classify a batch and attach reusable molecular graph metadata."""

    classified = []
    for row in rows:
        status, metadata = molecule_metadata(row.get("smiles", ""))
        if status == "kept" and metadata.molecule_size >= atom_limit:
            status = "too_large"
        classified.append((row, status, metadata))
    return classified


def _row_batches(
    rows: Iterator[dict[str, str]], batch_size: int
) -> Iterator[list[dict[str, str]]]:
    iterator = iter(rows)
    while batch := list(islice(iterator, batch_size)):
        yield batch


def _classified_rows(
    rows: Iterator[dict[str, str]],
    *,
    atom_limit: int,
    workers: int,
    worker_batch_size: int,
) -> Iterator[tuple[dict[str, str], str, MoleculeMetadata]]:
    """Classify with bounded, ordered multiprocessing and bounded memory."""

    batches = _row_batches(rows, worker_batch_size)
    if workers == 1:
        for batch in batches:
            yield from _classify_batch(batch, atom_limit)
        return

    # ProcessPoolExecutor.map eagerly consumes input on supported Python
    # versions. Maintain at most two batches per worker so PubChem-scale input
    # remains streaming rather than accumulating millions of pending futures.
    with ProcessPoolExecutor(
        max_workers=workers, initializer=_disable_rdkit_diagnostics
    ) as executor:
        pending: deque[Any] = deque()
        for _ in range(2 * workers):
            try:
                batch = next(batches)
            except StopIteration:
                break
            pending.append(executor.submit(_classify_batch, batch, atom_limit))
        while pending:
            yield from pending.popleft().result()
            try:
                batch = next(batches)
            except StopIteration:
                continue
            pending.append(executor.submit(_classify_batch, batch, atom_limit))


def _rows(path: Path) -> tuple[list[str], Iterator[dict[str, str]]]:
    """Read either a named CSV or PubChem's tab-separated ``CID SMILES`` file."""

    stream = path.open(newline="")
    first = stream.readline()
    stream.seek(0)
    if "\t" in first and "smiles" not in first.lower():
        fields = ["pubchem_id", "smiles"]

        def tab_rows() -> Iterator[dict[str, str]]:
            with stream:
                for line in stream:
                    identifier, separator, smiles = line.rstrip("\r\n").partition("\t")
                    if separator:
                        yield {"pubchem_id": identifier, "smiles": smiles}

        return fields, tab_rows()

    reader = csv.DictReader(stream)
    fields = list(reader.fieldnames or [])
    if "smiles" not in fields:
        stream.close()
        raise ValueError(f"{path} must contain a 'smiles' column.")

    def csv_rows() -> Iterator[dict[str, str]]:
        with stream:
            yield from reader

    return fields, csv_rows()


def _data_row_count(path: Path) -> int:
    """Count one-record-per-line input rows for a percentage/ETA display."""

    newline_count = 0
    last_byte = b""
    with path.open("rb") as stream:
        first = stream.readline()
        if not first:
            return 0
        has_header = b"\t" not in first or b"smiles" in first.lower()
        newline_count = first.count(b"\n")
        last_byte = first[-1:]
        while chunk := stream.read(8 * 1024 * 1024):
            newline_count += chunk.count(b"\n")
            last_byte = chunk[-1:]
    physical_lines = newline_count + (last_byte != b"\n")
    return max(0, physical_lines - int(has_header))


def default_output_path(input_path: Path, atom_limit: int) -> Path:
    suffix = f"_{atom_limit}.csv"
    stem = input_path.stem
    if stem.endswith("_raw"):
        stem = stem[:-4]
    return input_path.with_name(stem + suffix)


def preprocess_smiles(
    input_path: Path,
    output_path: Path,
    *,
    atom_limit: int = 32,
    seed: int = 0,
    validation_fraction: float = 0.05,
    test_fraction: float = 0.05,
    progress_every: int = 10_000,
    show_progress: bool = True,
    count_total: bool = True,
    workers: int | None = None,
    worker_batch_size: int = 2_000,
) -> Counter[str]:
    """Keep graph-buildable molecules below ``atom_limit`` heavy atoms."""

    if atom_limit < 2:
        raise ValueError("atom_limit must be at least 2.")
    if validation_fraction < 0 or test_fraction < 0:
        raise ValueError("split fractions cannot be negative.")
    if validation_fraction + test_fraction >= 1:
        raise ValueError("validation_fraction + test_fraction must be below 1.")
    if input_path.resolve() == output_path.resolve():
        raise ValueError("Input and output paths must differ.")
    if progress_every < 0:
        raise ValueError("progress_every cannot be negative.")
    if workers is None:
        workers = max(1, os.cpu_count() or 1)
    if workers < 1:
        raise ValueError("workers must be positive.")
    if worker_batch_size < 1:
        raise ValueError("worker_batch_size must be positive.")

    fields, rows = _rows(input_path)
    if set(FINGERPRINT_GRAPH_METADATA_COLUMNS) & set(fields):
        raise ValueError(
            "Input already contains Fingerprint2Graph metadata columns; "
            "provide a raw source file instead."
        )
    total_rows = _data_row_count(input_path) if show_progress and count_total else None
    output_path.parent.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    counts: Counter[str] = Counter()
    RDLogger.DisableLog("rdApp.error")
    RDLogger.DisableLog("rdApp.warning")
    try:
        with output_path.open("w", newline="") as stream:
            writer = csv.DictWriter(
                stream,
                fieldnames=fields + ["fold", *FINGERPRINT_GRAPH_METADATA_COLUMNS],
            )
            writer.writeheader()
            classified_rows = _classified_rows(
                rows,
                atom_limit=atom_limit,
                workers=workers,
                worker_batch_size=worker_batch_size,
            )
            with tqdm(
                classified_rows,
                desc=f"preprocessing {input_path.name}",
                total=total_rows,
                unit="rows",
                dynamic_ncols=True,
                disable=not show_progress,
            ) as progress:
                for row, status, metadata in progress:
                    counts["rows"] += 1
                    if status != "kept":
                        counts[status] += 1
                    else:
                        draw = float(rng.random())
                        if draw < test_fraction:
                            fold = "test"
                        elif draw < test_fraction + validation_fraction:
                            fold = "val"
                        else:
                            fold = "train"
                        writer.writerow({**row, "fold": fold, **metadata.as_csv_row()})
                        counts["kept"] += 1
                        counts[fold] += 1
                    if progress_every and counts["rows"] % progress_every == 0:
                        progress.set_postfix(
                            kept=counts["kept"],
                            too_large=counts["too_large"],
                            invalid=(
                                counts["invalid_smiles"]
                                + counts["empty_molecule"]
                                + counts["unsupported_bond"]
                                + counts["rdkit_error"]
                            ),
                            refresh=False,
                        )
    finally:
        RDLogger.EnableLog("rdApp.error")
        RDLogger.EnableLog("rdApp.warning")
    return counts


def has_fingerprint_graph_metadata(columns: Sequence[str]) -> bool:
    """Return whether all reusable graph-metadata columns are present."""

    return set(FINGERPRINT_GRAPH_METADATA_COLUMNS) <= set(columns)


def upgrade_fingerprint_graph_csv(
    path: Path,
    *,
    workers: int = 1,
    worker_batch_size: int = 2_000,
) -> Counter[str]:
    """Atomically add strict-graph metadata to an older preprocessed CSV.

    Rows and non-metadata columns are retained in order. Work is streamed
    through the same bounded RDKit process pool as preprocessing; the original
    CSV is replaced only after the temporary upgraded file closes successfully.
    """

    if workers < 1:
        workers = 1
    with path.open(newline="") as source:
        reader = csv.DictReader(source)
        fields = list(reader.fieldnames or [])
    if not {"smiles", "fold"} <= set(fields):
        raise ValueError(f"{path} must contain 'smiles' and 'fold' columns.")
    if has_fingerprint_graph_metadata(fields):
        return Counter()

    retained_fields = [
        field for field in fields if field not in FINGERPRINT_GRAPH_METADATA_COLUMNS
    ]
    destination_fields = [*retained_fields, *FINGERPRINT_GRAPH_METADATA_COLUMNS]
    descriptor, temporary_name = tempfile.mkstemp(
        dir=path.parent, prefix=f".{path.name}.", suffix=".metadata.tmp"
    )
    os.close(descriptor)
    temporary_path = Path(temporary_name)
    counts: Counter[str] = Counter()
    print(
        f"[data] adding Fingerprint2Graph graph metadata to {path} "
        f"with {workers} worker(s)...",
        flush=True,
    )
    try:
        with (
            path.open(newline="") as source,
            temporary_path.open("w", newline="") as destination,
        ):
            reader = csv.DictReader(source)
            writer = csv.DictWriter(destination, fieldnames=destination_fields)
            writer.writeheader()
            rows = (dict(row) for row in reader)
            classified = _classified_rows(
                rows,
                atom_limit=2**31 - 1,
                workers=workers,
                worker_batch_size=worker_batch_size,
            )
            with tqdm(
                classified,
                desc=f"upgrading {path.name}",
                unit="rows",
                dynamic_ncols=True,
            ) as progress:
                for row, status, metadata in progress:
                    counts["rows"] += 1
                    counts[status] += 1
                    writer.writerow(
                        {
                            **{field: row.get(field, "") for field in retained_fields},
                            **metadata.as_csv_row(),
                        }
                    )
        os.replace(temporary_path, path)
    except BaseException:
        temporary_path.unlink(missing_ok=True)
        raise
    print(f"[data] graph metadata ready: {dict(counts)}", flush=True)
    return counts


def argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Filter and split a Fingerprint2Graph raw SMILES file",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "input",
        type=Path,
        nargs="?",
        help=(
            "raw SMILES file; when omitted, acquire and preprocess the official "
            "PubChem CID--SMILES source"
        ),
    )
    parser.add_argument("--output", type=Path)
    parser.add_argument("--atom-limit", type=int, default=32)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--validation-fraction", type=float, default=0.05)
    parser.add_argument("--test-fraction", type=float, default=0.05)
    parser.add_argument(
        "--progress",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="display a tqdm row counter, processing rate, and filter counts",
    )
    parser.add_argument(
        "--count-total",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="scan line count first so tqdm can display percentage and ETA",
    )
    parser.add_argument(
        "--progress-every",
        type=int,
        default=10_000,
        help="rows between tqdm kept/removed count updates; 0 disables postfix updates",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=max(1, os.cpu_count() or 1),
        help="RDKit molecule-parsing worker processes; use 1 for serial execution",
    )
    parser.add_argument(
        "--worker-batch-size",
        type=int,
        default=2_000,
        help="rows sent to each worker per task",
    )
    return parser


def main(arguments: Sequence[str] | None = None) -> None:
    args = argument_parser().parse_args(arguments)
    input_path = args.input or ensure_pubchem_raw_dataset()
    output = args.output or default_output_path(input_path, args.atom_limit)
    counts = preprocess_smiles(
        input_path,
        output,
        atom_limit=args.atom_limit,
        seed=args.seed,
        validation_fraction=args.validation_fraction,
        test_fraction=args.test_fraction,
        progress_every=args.progress_every,
        show_progress=args.progress,
        count_total=args.count_total,
        workers=args.workers,
        worker_batch_size=args.worker_batch_size,
    )
    print(f"wrote {output}")
    print(dict(counts))


if __name__ == "__main__":
    main()
