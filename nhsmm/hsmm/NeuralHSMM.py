# nhsmm/hsmm/NeuralHSMM.py
from __future__ import annotations
from typing import Optional
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical, MultivariateNormal, Distribution
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
        dof = (self.n_states - 1) + self.n_states * (self.n_states - 1) + self.n_states * (self.max_duration - 1)
        pdf = self.pdf
        if pdf is not None:
            if isinstance(pdf, Categorical):
                dof += self.n_states * (pdf.logits.shape[1] - 1)
            elif isinstance(pdf, MultivariateNormal):
                dof += self.n_states * self.n_features * 2
        return dof

    # ----------------------
    # Emission PDFs
    # ----------------------
    def sample_emission_pdf(self, X: Optional[torch.Tensor] = None) -> Distribution:
        if self.emission_type == "categorical":
            logits = torch.ones((self.n_states, self.n_features), dtype=DTYPE, device=self.device)
            logits = logits / logits.sum(dim=1, keepdim=True)
            return Categorical(logits=torch.log(logits))
        elif self.emission_type == "gaussian":
            mean = torch.zeros(self.n_states, self.n_features, dtype=DTYPE, device=self.device)
            cov = torch.stack([torch.eye(self.n_features, dtype=DTYPE, device=self.device) * self.min_covar
                               for _ in range(self.n_states)])
            return MultivariateNormal(loc=mean, covariance_matrix=cov)
        else:
            raise ValueError(f"Unsupported emission_type '{self.emission_type}'")

    def _estimate_emission_pdf(self, X: torch.Tensor, posterior: torch.Tensor,
                               theta: Optional[torch.Tensor]) -> Distribution:
        if self.emission_type == "categorical":
            probs = (posterior.T @ X).to(dtype=DTYPE, device=self.device)
            probs = probs / (probs.sum(dim=1, keepdim=True) + 1e-12)
            return Categorical(probs=probs)
        elif self.emission_type == "gaussian":
            weights = posterior / (posterior.sum(dim=0, keepdim=True) + 1e-12)
            mean = weights.T @ X
            cov = torch.zeros(self.n_states, self.n_features, self.n_features, dtype=DTYPE, device=self.device)
            for k in range(self.n_states):
                diff = X - mean[k]
                weighted_diff = diff * weights[:, k].unsqueeze(-1)
                cov_k = weighted_diff.T @ diff
                cov[k] = cov_k + torch.eye(self.n_features, dtype=DTYPE, device=self.device) * self.min_covar
            return MultivariateNormal(loc=mean, covariance_matrix=cov)
        else:
            raise ValueError(f"Unsupported emission_type '{self.emission_type}'")

    # ----------------------
    # Contextual hooks
    # ----------------------
    def _contextual_emission_pdf(self, X: utils.Observations,
                                 theta: Optional[torch.Tensor]) -> Distribution:
        pdf = self.pdf
        if pdf is None or theta is None:
            return pdf
        theta_proj = theta.mean(dim=0)
        if isinstance(pdf, Categorical):
            delta = theta_proj[:pdf.logits.shape[1]]
            return Categorical(logits=pdf.logits + delta)
        elif isinstance(pdf, MultivariateNormal):
            mean_shift = theta_proj[:pdf.mean.shape[1]]
            return MultivariateNormal(loc=pdf.mean + mean_shift,
                                      covariance_matrix=pdf.covariance_matrix)
        return pdf

    def _contextual_transition_matrix(self, theta: Optional[torch.Tensor]) -> torch.Tensor:
        base_logits = self.transition_module.logits
        if theta is None:
            return F.softmax(base_logits, dim=-1)
        delta = theta.mean(dim=0)
        n = min(self.n_states, delta.shape[0])
        return F.softmax(base_logits + delta[:n].unsqueeze(0).expand(self.n_states, n), dim=-1)

    def _contextual_duration_pdf(self, theta: Optional[torch.Tensor]) -> torch.Tensor:
        base_logits = self.duration_module.logits
        if theta is None:
            return F.softmax(base_logits, dim=-1)
        delta = theta.mean(dim=0)
        n = min(self.max_duration, delta.shape[0])
        return F.softmax(base_logits + delta[:n].unsqueeze(0).expand(self.n_states, n), dim=-1)

    # ----------------------
    # Encoder forward
    # ----------------------
    def encode_observations(self, X: torch.Tensor) -> Optional[torch.Tensor]:
        if self.encoder is None:
            return None
        inp = X if X.ndim == 3 else X.unsqueeze(0)
        out = self.encoder(inp)
        if isinstance(out, tuple):
            out = out[0]
        out = out.detach().to(dtype=DTYPE, device=self.device)
        if out.ndim == 3:
            vec = out.mean(dim=1)
        elif out.ndim == 2:
            vec = out
        elif out.ndim == 1:
            vec = out.unsqueeze(0)
        else:
            raise ValueError(f"Unsupported encoder output shape {out.shape}")
        return vec

    # ----------------------
    # Forward / Predict
    # ----------------------
    def forward(self, X: torch.Tensor) -> torch.Tensor:
        return self.encode_observations(X)

    def predict(self, X: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        return super().predict(X, *args, **kwargs)
