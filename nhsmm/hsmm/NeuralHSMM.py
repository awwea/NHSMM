# nhsmm/hsmm/NeuralHSMM.py
from __future__ import annotations
from typing import Optional
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Distribution, Categorical, MultivariateNormal
from nhsmm.hsmm.BaseHSMM import BaseHSMM, DTYPE
from nhsmm.utilities import utils

# -----------------------------
# Default modules (pluggable)
# -----------------------------
class DefaultEmission(nn.Module):
    """Gaussian emissions per state (pluggable)."""
    def __init__(self, n_states, n_features, min_covar=1e-3):
        super().__init__()
        self.n_states = n_states
        self.n_features = n_features
        self.min_covar = min_covar
        self.mu = nn.Parameter(torch.zeros(n_states, n_features, dtype=DTYPE))
        self.log_var = nn.Parameter(torch.zeros(n_states, n_features, dtype=DTYPE))

    def forward(self, X=None):
        var = F.softplus(self.log_var) + self.min_covar
        return self.mu, var

class DefaultDuration(nn.Module):
    """Optional covariate-conditioned duration module (pluggable)."""
    def __init__(self, n_states, max_duration=20):
        super().__init__()
        self.n_states = n_states
        self.max_duration = max_duration
        self.logits = nn.Parameter(torch.zeros(n_states, max_duration, dtype=DTYPE))

    def forward(self, covariates=None):
        return F.softmax(self.logits, dim=-1)

class DefaultTransition(nn.Module):
    """Learnable transition matrix (pluggable)."""
    def __init__(self, n_states):
        super().__init__()
        self.n_states = n_states
        self.logits = nn.Parameter(torch.zeros(n_states, n_states, dtype=DTYPE))

    def forward(self):
        return F.softmax(self.logits, dim=-1)

# -----------------------------
# NeuralHSMM
# -----------------------------
class NeuralHSMM(BaseHSMM, nn.Module):
    """
    Hidden semi-Markov model with neural contextual encoder (CNN/LSTM/Transformer).

    Extends BaseHSMM:
    - Neural encoder extracts sequence-level context (`theta`) from X
    - Adapts emission, transition, and duration distributions based on context
    - EM, Viterbi, scoring remain fully compatible
    """

    def __init__(self,
                 n_states: int,
                 max_duration: int,
                 n_features: int,
                 alpha: float = 1.0,
                 seed: Optional[int] = None,
                 encoder: Optional[nn.Module] = None,
                 emission_type: str = "gaussian",
                 min_covar: float = 1e-3,
                 device: Optional[torch.device] = None):

        nn.Module.__init__(self)
        self.device = device or torch.device("cpu")
        self.n_features = n_features
        self.min_covar = min_covar
        self.encoder = encoder
        self._params = {'emission_type': emission_type.lower()}

        # Initialize BaseHSMM (required for EM, Viterbi, scoring)
        super().__init__(n_states=n_states, max_duration=max_duration, alpha=alpha, seed=seed)
        self._params['emission_pdf'] = self.sample_emission_pdf()

        # Default pluggable modules
        self.emission_module = DefaultEmission(n_states, n_features, min_covar)
        self.duration_module = DefaultDuration(n_states, max_duration)
        self.transition_module = DefaultTransition(n_states)

    # ----------------------
    # Properties
    # ----------------------
    @property
    def emission_type(self) -> str:
        return self._params.get('emission_type', 'gaussian')

    @property
    def pdf(self) -> Distribution:
        return self._params.get('emission_pdf', None)

    @property
    def dof(self) -> int:
        """
        Total degrees of freedom (number of learnable parameters)
        including initial, transition, duration, and emission parameters.
        Assumes diagonal covariance for Gaussian emissions.
        """
        nS, nD, nF = self.n_states, self.max_duration, self.n_features

        # Core HSMM parameters
        dof = (nS - 1)              # initial state probabilities (pi)
        dof += nS * (nS - 1)        # transition matrix (A)
        dof += nS * (nD - 1)        # duration distributions (D)

        # Emission parameters
        pdf = self.pdf
        if pdf is not None:
            if isinstance(pdf, Categorical):
                dof += nS * (pdf.logits.shape[1] - 1)
            elif isinstance(pdf, MultivariateNormal):
                # diagonal Gaussian: mean + variance per feature
                dof += nS * (2 * nF)

        return dof

    # ----------------------
    # Emission PDFs
    # ----------------------
    def sample_emission_pdf(self, X: Optional[torch.Tensor] = None) -> Distribution:
        """
        Initialize emission distribution for each hidden state.
        Supports 'categorical' and 'gaussian' types.

        If X is provided, can optionally be used to initialize parameters
        from empirical data (currently unused).
        """
        nS, nF, dev, dt = self.n_states, self.n_features, self.device, DTYPE

        if self.emission_type == "categorical":
            # Uniform categorical logits per state
            logits = torch.full((nS, nF), fill_value=1.0 / nF, dtype=dt, device=dev).log()
            return Categorical(logits=logits)

        elif self.emission_type == "gaussian":
            # Mean = 0, diagonal covariance = min_covar * I
            mean = torch.zeros(nS, nF, dtype=dt, device=dev)
            var = torch.full((nS, nF), self.min_covar, dtype=dt, device=dev)
            cov = torch.diag_embed(var)
            return MultivariateNormal(loc=mean, covariance_matrix=cov)

        else:
            raise ValueError(f"Unsupported emission_type '{self.emission_type}'")

    def _estimate_emission_pdf(
        self,
        X: torch.Tensor,
        posterior: torch.Tensor,
        theta: Optional[torch.Tensor] = None
    ) -> Distribution:
        """
        Estimate emission parameters (M-step) from data and posterior responsibilities.

        Parameters
        ----------
        X : (T, F) tensor
            Observations.
        posterior : (T, K) tensor
            State posterior probabilities.
        theta : Optional contextual tensor (unused for now)
        """
        K, F, dev, dt = self.n_states, self.n_features, self.device, DTYPE

        if self.emission_type == "categorical":
            # Assume X is one-hot encoded or counts
            if X.ndim != 2 or X.shape[1] != F:
                raise ValueError("For categorical emissions, X must be one-hot encoded with shape (T, n_features)")
            probs = posterior.T @ X
            probs = probs / (probs.sum(dim=1, keepdim=True) + 1e-12)
            return Categorical(probs=probs.clamp_min(1e-8))

        elif self.emission_type == "gaussian":
            # Normalized weights per state
            Nk = posterior.sum(dim=0, keepdim=True) + 1e-12
            weights = posterior / Nk
            mean = weights.T @ X  # (K, F)

            # Center and compute covariances
            diff = X.unsqueeze(1) - mean.unsqueeze(0)  # (T, K, F)
            w = posterior.unsqueeze(-1)
            cov = torch.einsum("tkf,tkh->kfh", w * diff, diff) / Nk.squeeze(0).unsqueeze(-1).unsqueeze(-1)

            # Regularize covariance for stability
            reg = self.min_covar * torch.eye(F, dtype=dt, device=dev).unsqueeze(0)
            cov = cov + reg

            return MultivariateNormal(loc=mean, covariance_matrix=cov)

        else:
            raise ValueError(f"Unsupported emission_type '{self.emission_type}'")

    # ----------------------
    # Contextual hooks
    # ----------------------
    def _contextual_emission_pdf(
        self,
        X: utils.Observations,
        theta: Optional[torch.Tensor]
    ) -> Distribution:
        """
        Modulate emission distribution using contextual features `theta`.
        Works for both Categorical and MultivariateNormal emissions.
        """
        pdf = self.pdf
        if pdf is None or theta is None:
            return pdf

        # Aggregate context across time if necessary
        theta_proj = theta.mean(dim=0) if theta.ndim > 1 else theta
        scale = 0.1  # scaling factor to prevent over-shift

        if isinstance(pdf, Categorical):
            # Ensure dimensional match
            n_classes = pdf.logits.shape[-1]
            delta = theta_proj[:n_classes].to(pdf.logits.device, pdf.logits.dtype)
            # Stabilize modification
            delta = scale * torch.tanh(delta)
            new_logits = pdf.logits + delta.unsqueeze(0).expand_as(pdf.logits)
            return Categorical(logits=new_logits)

        elif isinstance(pdf, MultivariateNormal):
            K, F = pdf.mean.shape
            mean_shift = theta_proj[:K * F].reshape(K, F).to(pdf.mean.device, pdf.mean.dtype)
            mean_shift = scale * mean_shift
            new_mean = pdf.mean + mean_shift
            return MultivariateNormal(loc=new_mean, covariance_matrix=pdf.covariance_matrix)

        return pdf

    def _contextual_transition_matrix(self, theta: Optional[torch.Tensor]) -> torch.Tensor:
        """
        Contextually modulate transition probabilities using latent features `theta`.
        Supports per-state adaptation via additive logit shifts.
        """
        base_logits = self.transition_module.logits
        if theta is None:
            return F.softmax(base_logits, dim=-1)

        # Aggregate context
        theta_proj = theta.mean(dim=0) if theta.ndim > 1 else theta
        theta_proj = theta_proj.to(base_logits.device, base_logits.dtype)

        # Project or pad context to full transition size
        n = self.n_states
        delta = theta_proj[: n * n].reshape(n, n)
        delta = 0.1 * torch.tanh(delta)  # Stabilize magnitude

        return F.softmax(base_logits + delta, dim=-1)

    def _contextual_duration_pdf(self, theta: Optional[torch.Tensor]) -> torch.Tensor:
        """
        Contextually modulate duration distributions based on latent features `theta`.
        Produces a per-state duration probability matrix.
        """
        base_logits = self.duration_module.logits
        if theta is None:
            return F.softmax(base_logits, dim=-1)

        theta_proj = theta.mean(dim=0) if theta.ndim > 1 else theta
        theta_proj = theta_proj.to(base_logits.device, base_logits.dtype)

        n_states, n_durations = base_logits.shape
        # Project or pad context to match full duration size
        delta = theta_proj[: n_states * n_durations].reshape(n_states, n_durations)
        delta = 0.1 * torch.tanh(delta)  # Stabilize scaling

        return F.softmax(base_logits + delta, dim=-1)

    # ----------------------
    # Encoder forward
    # ----------------------
    def encode_observations(self, X: torch.Tensor, detach: bool = True) -> Optional[torch.Tensor]:
        if self.encoder is None:
            return None

        inp = X if X.ndim == 3 else X.unsqueeze(0)
        out = self.encoder(inp)
        if isinstance(out, tuple):
            out = out[0]

        if detach:
            out = out.detach()

        out = out.to(dtype=DTYPE, device=self.device)

        if out.ndim == 3:  # (batch, seq_len, features)
            vec = out.mean(dim=1)
        elif out.ndim == 2:  # (batch, features)
            vec = out
        elif out.ndim == 1:  # (features,)
            vec = out.unsqueeze(0)
        else:
            raise ValueError(f"Unsupported encoder output shape {out.shape} (ndim={out.ndim})")

        return vec

    # ----------------------
    # Forward / Predict
    # ----------------------
    def forward(self, X: torch.Tensor, return_pdf: bool = False) -> torch.Tensor:
        theta = self.encode_observations(X)
        if return_pdf:
            return self._contextual_emission_pdf(X, theta)
        return theta

    def predict(self, X: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        return super().predict(X, *args, **kwargs)
