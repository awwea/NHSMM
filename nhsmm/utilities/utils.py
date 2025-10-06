from typing import Tuple, List, Optional, Union
from dataclasses import dataclass, field
import torch


@dataclass(frozen=False)
class Observations:
    """
    Container for observation sequences and their corresponding log-probabilities.
    Used as input/output for HSMM emission or inference modules.
    """
    sequence: List[torch.Tensor]
    log_probs: Optional[List[torch.Tensor]] = None
    lengths: Optional[List[int]] = None

    def __post_init__(self):
        if not self.sequence:
            raise ValueError("`sequence` cannot be empty")

        if not all(isinstance(s, torch.Tensor) for s in self.sequence):
            raise TypeError("All elements in `sequence` must be torch.Tensor")

        if self.log_probs is not None:
            if len(self.log_probs) != len(self.sequence):
                raise ValueError("`log_probs` length must match `sequence` length")
            if not all(isinstance(lp, torch.Tensor) for lp in self.log_probs):
                raise TypeError("All elements in `log_probs` must be torch.Tensor")

        lengths = self.lengths or [s.shape[0] for s in self.sequence]
        if any(s.shape[0] != l for s, l in zip(self.sequence, lengths)):
            raise ValueError("Mismatch between sequence lengths and provided `lengths`")
        object.__setattr__(self, "lengths", lengths)

    @property
    def n_sequences(self) -> int:
        """Number of independent observation sequences."""
        return len(self.sequence)

    @property
    def total_length(self) -> int:
        """Total number of time steps across all sequences."""
        return sum(self.lengths)

    @property
    def feature_dim(self) -> Optional[int]:
        """Return feature dimensionality if consistent across sequences."""
        dims = {s.shape[1] for s in self.sequence if s.ndim > 1}
        return dims.pop() if len(dims) == 1 else None

    def to(self, device: Union[str, torch.device]) -> "Observations":
        """Move all contained tensors to the specified device."""
        seqs = [s.to(device) for s in self.sequence]
        logs = [l.to(device) for l in self.log_probs] if self.log_probs else None
        return Observations(sequence=seqs, log_probs=logs, lengths=self.lengths)

    def pad_sequences(self, pad_value: float = 0.0) -> torch.Tensor:
        """
        Return zero-padded observation tensor (batch, T_max, features).
        Useful for batching variable-length sequences.
        """
        max_len = max(self.lengths)
        feature_dim = self.feature_dim or self.sequence[0].shape[-1]
        batch = torch.full((self.n_sequences, max_len, feature_dim), pad_value, dtype=self.sequence[0].dtype)
        for i, seq in enumerate(self.sequence):
            L = seq.shape[0]
            batch[i, :L] = seq
        return batch


@dataclass(frozen=True)
class ContextualVariables:
    """
    Represents exogenous or contextual features for a hierarchical/neural HSMM.
    Can encode embeddings, control signals, or time-varying covariates.
    """
    n_context: int
    X: Tuple[torch.Tensor, ...]
    time_dependent: bool = field(default=False)

    def __post_init__(self):
        if not self.X:
            raise ValueError("`X` cannot be empty")

        if not all(isinstance(x, torch.Tensor) for x in self.X):
            raise TypeError("All elements in `X` must be torch.Tensor")

        if len(self.X) != self.n_context:
            raise ValueError(f"`n_context` ({self.n_context}) must match number of tensors in `X` ({len(self.X)})")

        if self.time_dependent and any(x.ndim < 2 for x in self.X):
            raise ValueError("Time-dependent context must have at least 2D tensors (T, F)")

    @property
    def shape(self) -> Tuple[torch.Size, ...]:
        """Return the shape of each contextual tensor."""
        return tuple(x.shape for x in self.X)

    def cat(self, dim: int = -1) -> torch.Tensor:
        """Concatenate all context tensors along the specified dimension."""
        return torch.cat(self.X, dim=dim)

    def to(self, device: Union[str, torch.device]) -> "ContextualVariables":
        """Move context tensors to the specified device."""
        X = tuple(x.to(device) for x in self.X)
        return ContextualVariables(n_context=self.n_context, X=X, time_dependent=self.time_dependent)
