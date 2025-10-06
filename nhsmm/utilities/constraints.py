from typing import Tuple, Optional, Union
import torch
import numpy as np
from enum import Enum


class Transitions(Enum):
    SEMI = "semi"
    ERGODIC = "ergodic"
    LEFT_TO_RIGHT = "left-to-right"


class InformCriteria(Enum):
    AIC = "AIC"
    BIC = "BIC"
    HQC = "HQC"


class CovarianceType(Enum):
    FULL = "full"
    DIAG = "diag"
    TIED = "tied"
    SPHERICAL = "spherical"


# -------------------------------------------------------------------------
# Sampling utilities
# -------------------------------------------------------------------------
def sample_probs(prior: float, target_size: Union[Tuple[int, ...], torch.Size]) -> torch.Tensor:
    """Draw probabilities from a symmetric Dirichlet prior."""
    alphas = torch.full(target_size, prior, dtype=torch.float64)
    return torch.distributions.Dirichlet(alphas).sample()


def sample_A(prior: float, n_states: int, A_type: Transitions) -> torch.Tensor:
    """Sample transition matrix from Dirichlet prior with structure constraints."""
    probs = sample_probs(prior, (n_states, n_states))

    if A_type == Transitions.ERGODIC:
        pass
    elif A_type == Transitions.SEMI:
        probs.fill_diagonal_(0.0)
    elif A_type == Transitions.LEFT_TO_RIGHT:
        probs = torch.triu(probs)
        # ensure each row sums > 0
        zero_rows = probs.sum(-1) == 0
        if zero_rows.any():
            probs[zero_rows, zero_rows] = 1.0
    else:
        raise NotImplementedError(f"Unsupported Transition type: {A_type}")

    return probs / probs.sum(dim=-1, keepdim=True).clamp_min(1e-12)


# -------------------------------------------------------------------------
# Information criteria
# -------------------------------------------------------------------------
def compute_information_criteria(
    samples: int, log_likelihood: torch.Tensor, dof: int, criterion: InformCriteria
) -> torch.Tensor:
    """Compute AIC, BIC, or HQC given log-likelihood and degrees of freedom."""
    log_s = torch.log(torch.tensor(float(samples), dtype=torch.float64))
    if criterion == InformCriteria.AIC:
        penalty = 2.0 * dof
    elif criterion == InformCriteria.BIC:
        penalty = dof * log_s
    elif criterion == InformCriteria.HQC:
        penalty = 2.0 * dof * torch.log(log_s)
    else:
        raise ValueError(f"Invalid information criterion: {criterion.value}")
    return -2.0 * log_likelihood + penalty


# -------------------------------------------------------------------------
# Transition matrix validation
# -------------------------------------------------------------------------
def is_valid_A(probs: torch.Tensor, A_type: Transitions) -> bool:
    """Check if a transition matrix is valid under a given topology."""
    if not torch.isfinite(probs).all() or torch.any(probs < 0):
        return False

    row_sum_ok = torch.allclose(probs.sum(-1), torch.ones(probs.size(0), device=probs.device), atol=1e-6)
    if not row_sum_ok:
        return False

    if A_type == Transitions.ERGODIC:
        return True
    elif A_type == Transitions.SEMI:
        return torch.allclose(probs.diagonal(), torch.zeros_like(probs.diagonal()), atol=1e-6)
    elif A_type == Transitions.LEFT_TO_RIGHT:
        return bool((probs.tril(-1) == 0).all())
    else:
        raise NotImplementedError(f"Unsupported Transition type: {A_type}")


# -------------------------------------------------------------------------
# Log normalization
# -------------------------------------------------------------------------
def log_normalize(matrix: torch.Tensor, dim: Union[int, Tuple[int, ...]] = 1) -> torch.Tensor:
    """Return log-normalized tensor (log_probs with logsumexp(...)=0)."""
    return matrix - torch.logsumexp(matrix, dim=dim, keepdim=True)


# -------------------------------------------------------------------------
# Lambda & covariance validation
# -------------------------------------------------------------------------
def validate_lambdas(lambdas: torch.Tensor, n_states: int, n_features: int) -> torch.Tensor:
    """Validate Poisson/exponential rate parameters."""
    if lambdas.shape != (n_states, n_features):
        raise ValueError(f"Expected shape {(n_states, n_features)}, got {tuple(lambdas.shape)}")
    if not torch.isfinite(lambdas).all():
        raise ValueError("lambdas must not contain NaNs or infinities")
    if (lambdas <= 0).any():
        raise ValueError("lambdas must be strictly positive")
    return lambdas


def validate_covars(
    covars: torch.Tensor,
    covariance_type: CovarianceType,
    n_states: int,
    n_features: int,
    n_components: Optional[int] = None,
) -> torch.Tensor:
    """Validate covariance matrices across all supported types."""
    if covariance_type == CovarianceType.SPHERICAL:
        if covars.numel() != n_features:
            raise ValueError(f"'spherical' covars must have length {n_features}")
        if (covars <= 0).any():
            raise ValueError("'spherical' covars must be positive")
        return covars

    if covariance_type == CovarianceType.TIED:
        if covars.shape != (n_features, n_features):
            raise ValueError("'tied' covars must have shape (n_features, n_features)")
        _assert_spd(covars)
        return covars

    if covariance_type == CovarianceType.DIAG:
        if covars.shape != (n_states, n_features):
            raise ValueError("'diag' covars must have shape (n_states, n_features)")
        if (covars <= 0).any():
            raise ValueError("'diag' covars must be positive")
        return covars

    if covariance_type == CovarianceType.FULL:
        valid_shape = (n_states, n_features, n_features)
        if n_components:
            valid_shape = (n_states, n_components, n_features, n_features)
        if covars.shape != valid_shape:
            raise ValueError(f"'full' covars must have shape {valid_shape}")
        flat = covars.view(-1, n_features, n_features)
        for i, mat in enumerate(flat):
            _assert_spd(mat, label=f"Covariance {i}")
        return covars

    raise NotImplementedError(f"Unsupported covariance type: {covariance_type.value}")


def _assert_spd(matrix: torch.Tensor, label: str = "Matrix"):
    """Assert that a covariance matrix is symmetric positive-definite."""
    if not torch.allclose(matrix, matrix.T, atol=1e-6):
        raise ValueError(f"{label} is not symmetric")
    _, info = torch.linalg.cholesky_ex(matrix)
    if info.any():
        raise ValueError(f"{label} is not positive-definite")


# -------------------------------------------------------------------------
# Covariance initialization / expansion
# -------------------------------------------------------------------------
def init_covars(tied_cv: torch.Tensor, covariance_type: CovarianceType, n_states: int) -> torch.Tensor:
    """Expand a base covariance according to type."""
    if covariance_type == CovarianceType.SPHERICAL:
        return tied_cv.mean().expand(n_states)
    if covariance_type == CovarianceType.TIED:
        return tied_cv
    if covariance_type == CovarianceType.DIAG:
        return tied_cv.diag().unsqueeze(0).expand(n_states, -1)
    if covariance_type == CovarianceType.FULL:
        return tied_cv.unsqueeze(0).expand(n_states, -1, -1)
    raise NotImplementedError(f"Unsupported covariance type: {covariance_type.value}")


def fill_covars(
    covars: torch.Tensor,
    covariance_type: CovarianceType,
    n_states: int,
    n_features: int,
    n_components: Optional[int] = None,
) -> torch.Tensor:
    """Return full (n_states, n_features, n_features) covariance matrices."""
    if covariance_type == CovarianceType.FULL:
        return covars
    if covariance_type == CovarianceType.DIAG:
        return torch.diag_embed(covars)
    if covariance_type == CovarianceType.TIED:
        return covars.unsqueeze(0).expand(n_states, -1, -1)
    if covariance_type == CovarianceType.SPHERICAL:
        eye = torch.eye(n_features, dtype=covars.dtype, device=covars.device)
        return eye.unsqueeze(0) * covars.view(-1, 1, 1)
    raise NotImplementedError(f"Unsupported covariance type: {covariance_type.value}")
