"""Optional Laplacian eigenvector features for real target nodes."""

from __future__ import annotations

import torch
from torch import Tensor, nn


class LaplacianPositionalEncoding(nn.Module):
    """Compute sign-oriented nonzero Laplacian modes with one batched ``eigh``."""

    def __init__(self, n_components: int, eigenvalue_tolerance: float = 1e-6) -> None:
        super().__init__()
        if n_components < 1:
            raise ValueError("n_components must be positive.")
        if eigenvalue_tolerance <= 0:
            raise ValueError("eigenvalue_tolerance must be positive.")
        self.n_components = n_components
        self.eigenvalue_tolerance = eigenvalue_tolerance

    def forward(self, adjacency: Tensor, node_mask: Tensor) -> Tensor:
        """Return ``[batch, nodes, components]`` modes and zero all padded rows."""
        if adjacency.ndim != 3 or adjacency.shape[1] != adjacency.shape[2]:
            raise ValueError("adjacency must have shape [batch, nodes, nodes].")
        if node_mask.shape != adjacency.shape[:2] or node_mask.dtype != torch.bool:
            raise ValueError("node_mask must be bool [batch, nodes].")
        work_dtype = (
            torch.float64 if adjacency.dtype == torch.float64 else torch.float32
        )
        graph = adjacency.to(work_dtype)
        real_pairs = node_mask.unsqueeze(2) & node_mask.unsqueeze(1)
        graph = graph.masked_fill(~real_pairs, 0.0)
        graph = (graph + graph.transpose(1, 2)) * 0.5
        diagonal = torch.eye(
            graph.shape[1], dtype=torch.bool, device=graph.device
        ).unsqueeze(0)
        graph = graph.masked_fill(diagonal, 0.0)
        laplacian = torch.diag_embed(graph.sum(dim=-1)) - graph

        # Padding would otherwise add zero eigenvalues before the real graph
        # modes. A simple graph has lambda_max <= 2*(N-1), so move padding
        # dimensions strictly above the real spectrum before batched eigh.
        padding_eigenvalue = float(2 * adjacency.shape[1] + 1)
        laplacian = laplacian + torch.diag_embed(
            (~node_mask).to(work_dtype) * padding_eigenvalue
        )
        eigenvalues, eigenvectors = torch.linalg.eigh(laplacian)

        valid = (eigenvalues > self.eigenvalue_tolerance) & (
            eigenvalues < padding_eigenvalue - self.eigenvalue_tolerance
        )
        ranked = torch.where(valid, eigenvalues, torch.inf)
        selected_count = min(self.n_components, adjacency.shape[1])
        indices = ranked.argsort(dim=1)[:, :selected_count]
        vectors = eigenvectors.gather(
            2, indices.unsqueeze(1).expand(-1, adjacency.shape[1], -1)
        )
        selected_valid = valid.gather(1, indices)
        vectors = vectors * selected_valid.unsqueeze(1)

        pivots = vectors.abs().argmax(dim=1)
        signs = vectors.gather(1, pivots.unsqueeze(1)).squeeze(1).sign()
        signs = torch.where(signs == 0, torch.ones_like(signs), signs)
        vectors = vectors * signs.unsqueeze(1)
        if selected_count < self.n_components:
            vectors = torch.cat(
                (
                    vectors,
                    vectors.new_zeros(
                        vectors.shape[0],
                        vectors.shape[1],
                        self.n_components - selected_count,
                    ),
                ),
                dim=2,
            )
        return vectors.to(adjacency.dtype)
