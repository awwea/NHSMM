from typing import Optional
import torch
from torch import nn
from torch.distributions import Multinomial

from nhsmm.utilities import utils, constraints


class NeuralMultinomialDist(Multinomial):
    """
    Neural Multinomial emission distribution for discrete HSMM observations.

    Extends torch.distributions.Multinomial with optional neural context encoders
    for state-conditional emission logits.
    """

    def __init__(
        self,
        logits: torch.Tensor,
        trials: int = 1,
        encoder: Optional[nn.Module] = None,
        context_mode: str = "none",
    ):
        super().__init__(total_count=trials, logits=logits)
        self.encoder = encoder
        self.context_mode = context_mode
        if context_mode not in {"none", "temporal", "spatial"}:
            raise ValueError(f"Invalid context_mode '{context_mode}'")

    @property
    def dof(self) -> int:
        """Degrees of freedom = batch_size * (num_categories - 1)."""
        return int(torch.prod(torch.tensor(self.batch_shape))) * (self.event_shape[0] - 1)

    @classmethod
    def sample_emission_pdf(
        cls,
        trials: int,
        n_states: int,
        n_categories: int,
        alpha: float = 1.0,
        X: Optional[torch.Tensor] = None,
        encoder: Optional[nn.Module] = None,
        context_mode: str = "none",
    ) -> "NeuralMultinomialDist":
        """
        Sample or initialize emission logits (supports encoder injection).
        """
        eps = 1e-8
        if X is not None:
            if X.ndim > 1 and X.shape[-1] == n_categories:
                emission_freqs = X.sum(dim=0) / X.sum()
            else:
                emission_freqs = torch.bincount(X.long(), minlength=n_categories).float()
                emission_freqs /= emission_freqs.sum()
            emission_matrix = emission_freqs.clamp_min(eps).log().unsqueeze(0).expand(n_states, -1)
        else:
            probs = constraints.sample_probs(alpha, (n_states, n_categories))
            emission_matrix = probs.clamp_min(eps).log()

        return cls(
            logits=emission_matrix,
            trials=trials,
            encoder=encoder,
            context_mode=context_mode,
        )

    def estimate_emission_pdf(
        self,
        X: torch.Tensor,
        posterior: torch.Tensor,
        theta: Optional[utils.ContextualVariables] = None,
    ) -> "NeuralMultinomialDist":
        """Estimate emission probabilities with or without neural context."""
        if self.encoder is not None and theta is not None:
            context_logits = self._contextual_logits(X, theta)
            return NeuralMultinomialDist(
                logits=context_logits,
                trials=self.total_count,
                encoder=self.encoder,
                context_mode=self.context_mode,
            )

        new_B = self._compute_B(X, posterior)
        return NeuralMultinomialDist(
            logits=new_B.clamp_min(1e-8).log(),
            trials=self.total_count,
            encoder=self.encoder,
            context_mode=self.context_mode,
        )

    def _contextual_logits(self, X: torch.Tensor, theta: utils.ContextualVariables) -> torch.Tensor:
        """Compute emission logits using neural encoder and contextual variables."""
        if not callable(self.encoder):
            raise ValueError("Encoder must be a callable nn.Module.")

        encoded = self.encoder(*theta.X) if isinstance(theta.X, tuple) else self.encoder(theta.X)
        if encoded.shape[-1] != self.event_shape[0]:
            raise ValueError(f"Encoder output dim {encoded.shape[-1]} != event_shape {self.event_shape[0]}")

        return torch.log_softmax(encoded, dim=-1)

    def _compute_B(self, X: torch.Tensor, posterior: torch.Tensor) -> torch.Tensor:
        """Compute expected emission probabilities per hidden state."""
        weighted_counts = posterior.T @ X
        denom = weighted_counts.sum(dim=1, keepdim=True).clamp_min(1e-8)
        weighted_counts /= denom
        return weighted_counts
