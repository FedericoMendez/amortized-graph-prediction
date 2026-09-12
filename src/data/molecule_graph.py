"""On-the-fly RDKit molecule and Murcko-scaffold graph construction."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from rdkit import Chem
from rdkit import rdBase
from rdkit.Chem.Scaffolds import MurckoScaffold

from any2graph_v2.parameter import (
    VALID_ATOMS_LIST,
    VALID_BOND_TYPES,
    GraphParameters,
)

from .dense_data import DenseData


def _mol_from_smiles_quiet(smiles: str) -> Chem.Mol | None:
    """Parse a SMILES without emitting RDKit diagnostics for discarded rows.

    Large source datasets contain a small number of isolated explicit hydrogen
    atoms and malformed records.  Both are handled by the molecule policy
    below, so RDKit's process-level stderr messages are not actionable during
    training and can overwhelm scheduler logs.
    """

    with rdBase.BlockLogs():
        return Chem.MolFromSmiles(smiles)


@dataclass(frozen=True)
class ResolvedMolecule:
    """Policy-resolved RDKit target before dense tensor construction."""

    molecule: Chem.Mol
    canonical_smiles: str
    target_smiles: str
    scaffold_fallback: bool
    truncated: bool
    original_atoms: int
    target_atoms_before_truncation: int


@dataclass(frozen=True)
class MoleculeGraphResult:
    graph: DenseData
    canonical_smiles: str
    target_smiles: str
    scaffold_fallback: bool
    truncated: bool
    original_atoms: int
    target_atoms_before_truncation: int


def resolve_target_molecule(
    smiles: str, parameters: GraphParameters
) -> ResolvedMolecule | None:
    """Apply scaffold, fallback, filtering, and size policies without tensors."""

    parsed = _mol_from_smiles_quiet(smiles)
    if parsed is None:
        return None
    # RDKit leaves isolated explicit hydrogen atoms untouched. The subsequent
    # heavy-atom and supported-atom policies decide whether the result is usable.
    with rdBase.BlockLogs():
        parsed = Chem.RemoveHs(parsed)
    canonical_smiles = Chem.MolToSmiles(parsed, canonical=True)
    original = _mol_from_smiles_quiet(canonical_smiles)
    if original is None or original.GetNumHeavyAtoms() == 0:
        return None

    target = original
    scaffold_fallback = False
    if parameters.scaffold:
        scaffold = MurckoScaffold.GetScaffoldForMol(original)
        if scaffold.GetNumHeavyAtoms() == 0:
            if parameters.remove_invalid_molecules:
                return None
            scaffold_fallback = True
        else:
            scaffold_smiles = Chem.MolToSmiles(scaffold, canonical=True)
            target = _mol_from_smiles_quiet(scaffold_smiles)
            if target is None:
                return None

    target_smiles = Chem.MolToSmiles(target, canonical=True)
    n_target = target.GetNumHeavyAtoms()
    supported_atoms = set(VALID_ATOMS_LIST) - {"UNK"}
    if parameters.remove_invalid_molecules and any(
        atom.GetSymbol() not in supported_atoms for atom in target.GetAtoms()
    ):
        return None
    if any(
        str(bond.GetBondType()) not in VALID_BOND_TYPES
        for bond in target.GetBonds()
    ):
        return None
    truncated = n_target > parameters.n_nodes_max
    if truncated and parameters.remove_invalid_molecules:
        return None

    return ResolvedMolecule(
        molecule=target,
        canonical_smiles=canonical_smiles,
        target_smiles=target_smiles,
        scaffold_fallback=scaffold_fallback,
        truncated=truncated,
        original_atoms=original.GetNumHeavyAtoms(),
        target_atoms_before_truncation=n_target,
    )


def build_molecular_graph(
    smiles: str, parameters: GraphParameters
) -> MoleculeGraphResult | None:
    """Build a dense graph, retaining the first canonical atoms when truncated."""

    resolved = resolve_target_molecule(smiles, parameters)
    if resolved is None:
        return None

    atom_to_index = {label: index for index, label in enumerate(VALID_ATOMS_LIST)}
    unknown_index = atom_to_index["UNK"]
    bond_to_index = {
        label: index + 1 for index, label in enumerate(VALID_BOND_TYPES)
    }
    size = min(resolved.target_atoms_before_truncation, parameters.n_nodes_max)

    node_indices = torch.empty(size, dtype=torch.long)
    for index, atom in enumerate(resolved.molecule.GetAtoms()):
        if index >= size:
            break
        symbol = atom.GetSymbol()
        if symbol not in atom_to_index and parameters.remove_invalid_molecules:
            return None
        node_indices[index] = atom_to_index.get(symbol, unknown_index)
    node_labels = torch.nn.functional.one_hot(
        node_indices, num_classes=len(VALID_ATOMS_LIST)
    ).to(torch.float32)

    edge_indices = torch.zeros((size, size), dtype=torch.long)
    adjacency = torch.zeros((size, size), dtype=torch.float32)
    for bond in resolved.molecule.GetBonds():
        source = bond.GetBeginAtomIdx()
        target = bond.GetEndAtomIdx()
        if source >= size or target >= size:
            continue
        label = str(bond.GetBondType())
        if label not in bond_to_index:
            return None
        edge_indices[source, target] = edge_indices[target, source] = bond_to_index[label]
        adjacency[source, target] = adjacency[target, source] = 1.0

    edge_labels = torch.nn.functional.one_hot(
        edge_indices, num_classes=1 + len(VALID_BOND_TYPES)
    ).to(torch.float32)
    graph = DenseData(
        size=size,
        h=torch.ones(size, dtype=torch.bool),
        nodes={"labels": node_labels},
        edges={"labels": edge_labels, "adjacency": adjacency},
    )
    return MoleculeGraphResult(
        graph=graph,
        canonical_smiles=resolved.canonical_smiles,
        target_smiles=resolved.target_smiles,
        scaffold_fallback=resolved.scaffold_fallback,
        truncated=resolved.truncated,
        original_atoms=resolved.original_atoms,
        target_atoms_before_truncation=resolved.target_atoms_before_truncation,
    )
