from typing import Tuple, List, Optional, Union
from dataclasses import dataclass, field
import torch


@dataclass
class Observations:
    """
    Encapsulates observation sequences and their corresponding log-probabilities.
    Typically used as the input/output of HSMM emission or inference steps.
    """
    sequence: List[torch.Tensor]
    log_probs: Optional[List[torch.Tensor]] = None
    lengths: Optional[List[int]] = None

    def __post_init__(self):
        # Basic sanity checks
        if not all(isinstance(seq, torch.Tensor) for seq in self.sequence):
            raise TypeError("All elements in `sequence` must be torch.Tensor")

        if self.log_probs is not None and len(self.log_probs) != len(self.sequence):
            raise ValueError("`log_probs` length must match `sequence` length")

        if self.lengths is None:
            self.lengths = [s.shape[0] for s in self.sequence]

    @property
    def n_sequences(self) -> int:
        """Return the number of observation sequences."""
        return len(self.sequence)

    @property
    def total_length(self) -> int:
        """Return total number of observations across all sequences."""
        return sum(self.lengths)

    def to(self, device: Union[str, torch.device]) -> "Observations":
        """Move all tensors to a specific device."""
        seqs = [s.to(device) for s in self.sequence]
        logs = [l.to(device) for l in self.log_probs] if self.log_probs is not None else None
        return Observations(sequence=seqs, log_probs=logs, lengths=self.lengths)


@dataclass
class ContextualVariables:
    """
    Represents additional context used by a hierarchical or neural HSMM.
    For example, exogenous features, embeddings, or time-varying covariates.
    """
    n_context: int
    X: Tuple[torch.Tensor, ...]
    time_dependent: bool = field(default=False)

    def __post_init__(self):
        if not all(isinstance(x, torch.Tensor) for x in self.X):
            raise TypeError("All elements in `X` must be torch.Tensor")

        if len(self.X) != self.n_context:
            raise ValueError("`n_context` must match number of tensors in `X`")

    def to(self, device: Union[str, torch.device]) -> "ContextualVariables":
        """Move context tensors to the target device."""
        return ContextualVariables(
            n_context=self.n_context,
            X=tuple(x.to(device) for x in self.X),
            time_dependent=self.time_dependent
        )
