"""Mass-spectrum feature transforms."""

from __future__ import annotations

import re

import numpy as np


CHEMICAL_ELEMENTS = (
    "H",
    "C",
    "O",
    "N",
    "P",
    "S",
    "Cl",
    "F",
    "Br",
    "I",
    "B",
    "As",
    "Si",
    "Se",
)
ELEMENT_SCALES = np.asarray(
    (102, 59, 25, 13, 3, 6, 6, 17, 4, 4, 1, 1, 5, 2), dtype=np.float32
)
FORMULA_TOKEN = re.compile(r"([A-Z][a-z]?)(\d*)")


class SubformulaTransform:
    """Port of the reference FLARE-compatible 16-feature transform."""

    feature_dim = 16

    @staticmethod
    def _counts(formula: str) -> dict[str, int]:
        counts = {element: 0 for element in CHEMICAL_ELEMENTS}
        for symbol, count in FORMULA_TOKEN.findall(formula):
            if symbol in counts:
                counts[symbol] += int(count) if count else 1
        return counts

    def __call__(
        self,
        mz: list[float] | None,
        intensities: list[float] | None,
        subformulas: list[str] | None,
    ) -> np.ndarray:
        if mz is None or intensities is None or subformulas is None:
            return np.zeros((1, self.feature_dim), dtype=np.float32)
        if not (len(mz) == len(intensities) == len(subformulas)):
            raise ValueError("Annotated spectrum fields have different lengths.")
        if not mz:
            return np.zeros((1, self.feature_dim), dtype=np.float32)
        tokens = np.zeros((len(mz), self.feature_dim), dtype=np.float32)
        tokens[:, 0] = np.asarray(mz, dtype=np.float32) / 1000.0
        tokens[:, 1] = np.asarray(intensities, dtype=np.float32)
        for row, formula in enumerate(subformulas):
            counts = self._counts(formula)
            tokens[row, 2:] = [counts[element] for element in CHEMICAL_ELEMENTS]
        tokens[:, 2:] /= ELEMENT_SCALES
        return tokens


def trim_raw_spectrum(spectrum: np.ndarray) -> np.ndarray:
    """Remove zero-padded raw peaks while retaining one token for empty input."""

    if spectrum.ndim != 2 or spectrum.shape[1] != 2:
        raise ValueError("A raw spectrum must have shape [peaks, 2].")
    real = np.any(spectrum != 0, axis=1)
    trimmed = np.asarray(spectrum[real], dtype=np.float32)
    if len(trimmed) == 0:
        return np.zeros((1, 2), dtype=np.float32)
    return trimmed

