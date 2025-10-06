# nhsmm/utilities/constraints.py
from typing import Tuple, Optional, Union
import numpy as np
import torch
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


# -----------------------------------------------------------------------------
# Sampling utilities
# -----------------------------------------------------------------------------
def sample_probs(prior: float, target_size: Union[Tuple[int, ...], torch.Size]) -> torch.Tensor:
    """Initialize a tensor of probabilities from a Dirichlet prior."""
    alphas = torch.full(size=target_size, fill_value=prior, dtype=torch.float64)
    return torch.distributions.Dirichlet(alphas).sample()


def sample_A(prior: float, n_states: int, A_type: Transitions) -> torch.Tensor:
    """Initialize transition matrix from Dirichlet distribution."""
    probs = sample_probs(prior, (n_states, n_states))
    if A_type == Transitions.ERGODIC:
        pass  # fully connected
    elif A_type == Transitions.SEMI:
        probs.fill_diagonal_(0.0)
    elif A_type == Transitions.LEFT_TO_RIGHT:
        probs = torch.triu(probs)
    else:
        raise NotImplementedError(f"Unsupported Transition matrix type: {A_type}")

    row_sum = probs.sum(dim=-1, keepdim=True).clamp_min(1e-12)
    probs /= row_sum
    return probs


# -----------------------------------------------------------------------------
# Model criteria
# -----------------------------------------------------------------------------
def compute_information_criteria(
    samples: int, log_likelihood: torch.Tensor, dof: int, criterion: InformCriteria
) -> torch.Tensor:
    """Compute model selection criterion (AIC/BIC/HQC) from log-likelihood."""
    if criterion == InformCriteria.AIC:
        penalty = 2.0 * dof
    elif criterion == InformCriteria.BIC:
        penalty = dof * np.log(samples)
    elif criterion == InformCriteria.HQC:
        penalty = 2.0 * dof * np.log(np.log(samples))
    else:
        raise NotImplementedError(f"Invalid information criterion: {criterion}")

    return -2.0 * log_likelihood + penalty


# -----------------------------------------------------------------------------
# Transition matrix validation
# -----------------------------------------------------------------------------
def is_valid_A(logits: torch.Tensor, A_type: Transitions) -> bool:
    """Check the validity of transition matrix given its type."""
    probs = logits.exp()
    if A_type == Transitions.ERGODIC:
        return bool(torch.all(probs > 0.0))
    elif A_type == Transitions.SEMI:
        return bool(torch.allclose(probs.diagonal(), torch.zeros_like(probs.diagonal())))
    elif A_type == Transitions.LEFT_TO_RIGHT:
        return bool(torch.all(probs.tril(-1) == 0))
    else:
        raise NotImplementedError(f"Unsupported Transition matrix type: {A_type}")


# -----------------------------------------------------------------------------
# Log normalization
# -----------------------------------------------------------------------------
def log_normalize(matrix: torch.Tensor, dim: Union[int, Tuple[int, ...]] = 1) -> torch.Tensor:
    """Return log-normalized tensor: log_probs such that logsumexp(...)=0."""
    return matrix - matrix.logsumexp(dim, keepdim=True)


# -----------------------------------------------------------------------------
# Lambda & covariance validation
# -----------------------------------------------------------------------------
def validate_lambdas(lambdas: torch.Tensor, n_states: int, n_features: int) -> torch.Tensor:
    """Validate Poisson/exponential rate parameters."""
    if lambdas.shape != (n_states, n_features):
        raise ValueError(f"Expected shape {(n_states, n_features)}, got {lambdas.shape}")
    if torch.any(torch.isnan(lambdas)) or torch.any(torch.isinf(lambdas)):
        raise ValueError("lambdas must not contain NaNs or infinities")
    if torch.any(lambdas <= 0):
        raise ValueError("lambdas must be positive")
    return lambdas


def validate_covars(
    covars: torch.Tensor,
    covariance_type: CovarianceType,
    n_states: int,
    n_features: int,
    n_components: Optional[int] = None,
) -> torch.Tensor:
    """Validate covariance tensors."""
    if n_components is None:
        valid_shape = torch.Size((n_states, n_features, n_features))
    else:
        valid_shape = torch.Size((n_states, n_components, n_features, n_features))

    if covariance_type == CovarianceType.SPHERICAL:
        if covars.numel() != n_features:
            raise ValueError("'spherical' covars must have length n_features")
        if torch.any(covars <= 0):
            raise ValueError("'spherical' covars must be positive")

    elif covariance_type == CovarianceType.TIED:
        if covars.shape[0] != covars.shape[1]:
            raise ValueError("'tied' covars must have shape (n_dim, n_dim)")
        eig_vals = torch.linalg.eigvalsh(covars)
        if not torch.allclose(covars, covars.T) or torch.any(eig_vals <= 0):
            raise ValueError("'tied' covars must be symmetric, positive-definite")

    elif covariance_type == CovarianceType.DIAG:
        if covars.ndim != 2 or covars.shape[1] != n_features:
            raise ValueError("'diag' covars must have shape (n_states, n_dim)")
        if torch.any(covars <= 0):
            raise ValueError("'diag' covars must be positive")

    elif covariance_type == CovarianceType.FULL:
        if covars.shape != valid_shape:
            raise ValueError(f"'full' covars must have shape {valid_shape}")
        for i, cv in enumerate(covars.view(-1, n_features, n_features)):
            eig_vals = torch.linalg.eigvalsh(cv)
            if not torch.allclose(cv, cv.T) or torch.any(eig_vals <= 0):
                raise ValueError(f"Covariance {i} is not symmetric, positive-definite")

    else:
        raise NotImplementedError(f"Unsupported covariance type: {covariance_type}")

    return covars


# -----------------------------------------------------------------------------
# Covariance initialization / fill
# -----------------------------------------------------------------------------
def init_covars(tied_cv: torch.Tensor, covariance_type: CovarianceType, n_states: int) -> torch.Tensor:
    """Initialize covariance matrices according to type."""
    if covariance_type == CovarianceType.SPHERICAL:
        return tied_cv.mean() * torch.ones((n_states,), dtype=tied_cv.dtype)
    elif covariance_type == CovarianceType.TIED:
        return tied_cv
    elif covariance_type == CovarianceType.DIAG:
        return tied_cv.diag().unsqueeze(0).expand(n_states, -1)
    elif covariance_type == CovarianceType.FULL:
        return tied_cv.unsqueeze(0).expand(n_states, -1, -1)
    else:
        raise NotImplementedError(f"Unsupported covariance type: {covariance_type}")


def fill_covars(
    covars: torch.Tensor,
    covariance_type: CovarianceType,
    n_states: int,
    n_features: int,
    n_components: Optional[int] = None,
) -> torch.Tensor:
    """Expand lower-rank covariance forms to full matrices."""
    if covariance_type == CovarianceType.FULL:
        return covars
    elif covariance_type == CovarianceType.DIAG:
        return torch.diag_embed(covars)
    elif covariance_type == CovarianceType.TIED:
        return covars.unsqueeze(0).expand(n_states, -1, -1)
    elif covariance_type == CovarianceType.SPHERICAL:
        eye = torch.eye(n_features, dtype=covars.dtype, device=covars.device)
        return eye.unsqueeze(0) * covars.unsqueeze(-1).unsqueeze(-1)
    else:
        raise NotImplementedError(f"Unsupported covariance type: {covariance_type}")
