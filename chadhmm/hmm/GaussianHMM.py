import torch
from typing import Optional, Literal
from .BaseHMM import BaseHMM # type: ignore
from ..emissions.gaussian_mix import GaussianMixtureEmissions # type: ignore
from ..emissions import GaussianEmissions # type: ignore
from ..utils import validate_sequence # type: ignore


class GaussianHMM(BaseHMM, GaussianEmissions):
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
                 n_states: int,
                 n_features: int,
                 k_means: bool = False,
                 alpha: float = 1.0,
                 covariance_type: COVAR_TYPES = 'full',
                 min_covar: float = 1e-3,
                 seed: Optional[int] = None,
                 device: Optional[torch.device] = None):

        BaseHMM.__init__(self,n_states,alpha,seed,device)
        
        GaussianEmissions.__init__(self,n_states,n_features,k_means,covariance_type,min_covar,device)

    @property
    def n_fit_params(self):
        """Return the number of trainable model parameters."""
        return {
            'states': self.n_states,
            'transitions': self.n_states**2,
            'means': self.n_states * self.n_features,
            'covars': {
                'spherical': self.n_states,
                'diag': self.n_states * self.n_features,
                'full': self.n_states * self.n_features * (self.n_features + 1) // 2,
                'tied': self.n_features * (self.n_features + 1) // 2,
            }[self.covariance_type],
        }
    
    @property
    def params(self):
        return {'pi': self.initial_vector.logits,
                'A': self.transition_matrix.logits,
                'means': self.means, 
                'covars': self.covs}

    @property
    def dof(self):
        return self.n_states**2 - 1 + self.means.numel() + self.covs.numel() 

    def _update_B_params(self,X,log_gamma,theta):
        gamma = [torch.exp(gamma) for gamma in log_gamma]
        GaussianEmissions.update_emission_params(self,X,gamma,theta)

    def check_sequence(self,sequence):
        return validate_sequence(sequence,False,self.n_features)
    
    def map_emission(self,x):
        return GaussianEmissions.map_emission(self,x)

    def sample_B_params(self,X,seed=None):
        self.means, self.covs = GaussianEmissions.sample_emission_params(self,X,seed)


class GaussianMixtureHMM(BaseHMM, GaussianMixtureEmissions):
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
                 seed: Optional[int] = None,
                 device: Optional[torch.device] = None):

        BaseHMM.__init__(self,n_states,alpha,seed,device)
        
        GaussianMixtureEmissions.__init__(self,n_states,n_features,n_components,k_means,alpha,covariance_type, min_covar,seed,device)

    @property
    def n_fit_params(self):
        """Return the number of trainable model parameters."""
        fit_params_dict = {
            'states': self.n_states,
            'transitions': self.n_states**2,
            'weights': self.n_states * self.n_components,
            'means': self.n_states * self.n_features * self.n_components,
            'covars': {
                'spherical': self.n_states,
                'diag': self.n_states * self.n_features,
                'full': self.n_states * self.n_features * (self.n_features + 1) // 2,
                'tied': self.n_features * (self.n_features + 1) // 2,
            }[self.covariance_type],
        }

        return fit_params_dict
    
    @property
    def params(self):
        return {'pi': self.initial_vector.logits,
                'A': self.transition_matrix.logits,
                'weights': self.weights.logits,
                'means': self.means, 
                'covars': self.covs}

    @property
    def dof(self):
        return self.n_states**2 - 1 + self.n_states*self.n_components - self.n_states + self.means.numel() + self.covs.numel() 

    def _update_B_params(self,X,log_gamma,theta=None):
        posterior_vec = []
        resp_vec = GaussianMixtureEmissions._compute_responsibilities(self,X)
        for resp,gamma_val in zip(resp_vec,log_gamma):
            posterior_vec.append(torch.exp(resp + gamma_val.T.unsqueeze(1)))

        GaussianMixtureEmissions.update_emission_params(self,X,posterior_vec,theta)

    def check_sequence(self,sequence):
        return validate_sequence(sequence,False,self.n_features)
    
    def map_emission(self,x):
        return GaussianMixtureEmissions.map_emission(self,x)

    def sample_B_params(self,X,seed=None):
        self._means, self._covs = GaussianMixtureEmissions.sample_emission_params(self,X,seed)