import torch
import torch.nn as nn
from typing import Optional, Literal

from .BaseHMM import BaseHMM # type: ignore
from ..emissions import GaussianEmissions, GaussianMixtureEmissions # type: ignore


class GaussianHMM(BaseHMM):
    """
    Gaussian Hidden Markov Model (Gaussian HMM)
    ----------
    This model assumes that the data follows a multivariate Gaussian distribution. 
    The model parameters (initial state probabilities, transition probabilities, emission means, and emission covariances) are learned using the Baum-Welch algorithm.

    Parameters:
    ----------
    n_states (int):
        Number of hidden states in the model.
    n_features (int):
        Number of features in the emission data.
    alpha (float):
        Dirichlet concentration parameter for the prior over initial state probabilities and transition probabilities.
    covariance_type (COVAR_TYPES):
        Type of covariance parameters to use for the emission distributions.
    min_covar (float):
        Floor value for covariance matrices.
    seed (Optional[int]):
        Random seed to use for reproducible results.
    """

    COVAR_TYPES = Literal['spherical', 'tied', 'diag', 'full']

    def __init__(self,
                 n_states:int,
                 n_features:int,
                 k_means:bool = False,
                 alpha:float = 1.0,
                 covariance_type:COVAR_TYPES = 'full',
                 min_covar:float = 1e-3,
                 seed:Optional[int] = None):

        BaseHMM.__init__(self,n_states,alpha,seed)
        self.add_module('emissions',
                        GaussianEmissions(n_states,n_features,k_means,covariance_type,min_covar))

    @property
    def dof(self):
        return self.n_states**2 - 1 + self.emissions.means.numel() + self.emissions.covs.numel() 

    def _update_B_params(self,X,log_gamma,theta):
        gamma = [torch.exp(gamma) for gamma in log_gamma]
        self.emissions.update_emission_params(X,gamma,theta)

    def check_sequence(self,X):
        return self.emissions.check_constraints(X)
    
    def map_emission(self,x):
        return self.emissions.map_emission(x)

    def sample_B_params(self,X=None):
        sampled_params = self.emissions.sample_emission_params(X)
        self.means:nn.Parameter = sampled_params.get('means')
        self.covs:nn.Parameter = sampled_params.get('covs')


class GaussianMixtureHMM(BaseHMM):
    """
    Gaussian Hidden Markov Model (Gaussian HMM)
    ----------
    This model assumes that the data follows a multivariate Gaussian distribution. 
    The model parameters (initial state probabilities, transition probabilities, emission means, and emission covariances) are learned using the Baum-Welch algorithm.

    Parameters:
    ----------
    n_states (int):
        Number of hidden states in the model.
    n_features (int):
        Number of features in the emission data.
    n_components (int):
        Number of components in the Gaussian mixture model.
    alpha (float):
        Dirichlet concentration parameter for the prior over initial state probabilities and transition probabilities.
    covariance_type (COVAR_TYPES):
        Type of covariance parameters to use for the emission distributions.
    min_covar (float):
        Floor value for covariance matrices.
    seed (Optional[int]):
        Random seed to use for reproducible results.
    """

    COVAR_TYPES = Literal['spherical', 'tied', 'diag', 'full']

    def __init__(self,
                 n_states: int,
                 n_features: int,
                 n_components: int = 1,
                 k_means: bool = False,
                 alpha: float = 1.0,
                 covariance_type: COVAR_TYPES = 'full',
                 min_covar: float = 1e-3,
                 seed: Optional[int] = None):

        BaseHMM.__init__(self,n_states,alpha,seed)
        self.add_module('emissions',
                        GaussianMixtureEmissions(n_states,n_features,n_components,k_means,alpha,covariance_type,min_covar,seed))

    @property
    def dof(self):
        return self.n_states**2 - 1 + self.n_states*self.emissions.n_components - self.n_states + self.emissions.means.numel() + self.emissions.covs.numel() 

    def _update_B_params(self,X,log_gamma,theta=None):
        posterior_vec = []
        resp_vec = self.emissions._compute_responsibilities(X)
        for resp,gamma_val in zip(resp_vec,log_gamma):
            posterior_vec.append(torch.exp(resp + gamma_val.T.unsqueeze(1)))

        self.emissions.update_emission_params(X,posterior_vec,theta)

    def check_sequence(self,X):
        return self.emissions.check_constraints(X)
    
    def map_emission(self,x):
        return self.emissions.map_emission(x)

    def sample_B_params(self,X=None):
        self._means, self._covs = self.emissions.sample_emission_params(X)