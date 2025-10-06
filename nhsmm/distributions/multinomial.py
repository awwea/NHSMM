from typing import Optional, Callable
import torch
from torch import nn
from torch.distributions import Multinomial

from nhsmm.utilities import utils, constraints


class NeuralMultinomialDist(Multinomial):
    """
    Neural Multinomial emission distribution for discrete HSMM observations.
    Supports contextual (neural) emission logits via CNN/LSTM encoders.

    Parameters
    ----------
    logits : torch.Tensor
        Logits defining the categorical probabilities per hidden state.
    trials : int, default=1
        Number of trials for multinomial sampling.
    encoder : Optional[nn.Module], default=None
        Optional neural encoder (e.g., CNN, LSTM, Transformer)
        that produces state-dependent logits given contextual input.
    context_mode : str, default="none"
        One of {"none", "temporal", "spatial"} defining context type.
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

    @property
    def dof(self) -> int:
        """Degrees of freedom = batch_size * (num_categories - 1)."""
        return int(torch.prod(torch.tensor(self.batch_shape))) * (self.event_shape[0] - 1)

    @classmethod
    def sample_emission_pdf(
        cls,
        trials: int,
        batch_shape: int,
        event_shape: int,
        alpha: float = 1.0,
        X: Optional[torch.Tensor] = None,
        encoder: Optional[nn.Module] = None,
        context_mode: str = "none",
    ):
        """Sample or initialize emission logits (supports encoder injection)."""
        eps = 1e-8
        if X is not None:
            if X.ndim > 1 and X.shape[-1] == event_shape:
                emission_freqs = X.sum(dim=0) / X.sum()
            else:
                emission_freqs = torch.bincount(X.long(), minlength=event_shape).float()
                emission_freqs /= emission_freqs.sum()
            emission_matrix = emission_freqs.clamp_min(eps).log().unsqueeze(0).expand(batch_shape, -1)
        else:
            probs = constraints.sample_probs(alpha, (batch_shape, event_shape))
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
        """
        Compute emission logits using neural encoder and contextual variables.
        Expected: theta.X is tuple of tensors, matching encoder input format.
        """
        if not callable(self.encoder):
            raise ValueError("Encoder must be a callable nn.Module.")

        # Feed contextual data into neural encoder
        if isinstance(theta.X, tuple):
            encoded = self.encoder(*theta.X)
        else:
            encoded = self.encoder(theta.X)

        # Output normalization
        logits = torch.log_softmax(encoded, dim=-1)
        return logits

    def _compute_B(
        self,
        X: torch.Tensor,
        posterior: torch.Tensor,
    ) -> torch.Tensor:
        """Compute expected emission probabilities per hidden state."""
        weighted_counts = posterior.T @ X
        weighted_counts /= weighted_counts.sum(dim=1, keepdim=True)
        return weighted_counts
