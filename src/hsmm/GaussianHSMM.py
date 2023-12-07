from typing import Optional, Literal
import torch

from .BaseHSMM import BaseHSMM # type: ignore
from ..emissions.mixtures import GaussianMixtureEmissions # type: ignore
from ..emissions import GaussianEmissions # type: ignore
from ..utils import validate_sequence # type: ignore


class GaussianHSMM(BaseHSMM, GaussianEmissions):
    """
    Gaussian Hidden Semi-Markov Model (Gaussian HSMM)
    ----------
    This model assumes that the data follows a multivariate Gaussian distribution. 
    The model parameters (initial state probabilities, transition probabilities, duration probabilities,emission means, and emission covariances) 
    are learned using the Baum-Welch algorithm.

    Parameters:
    ----------
    n_states (int):
        Number of hidden states in the model.
    max_duration (int):
        Maximum duration of the states.
    n_features (int):
        Number of features in the emission data.
    n_components (int):
        Number of components in the Gaussian mixture model.
    params_init (bool):
        Whether to initialize the model parameters prior to fitting.
    alpha (float):
        Dirichlet concentration parameter for the prior over initial state probabilities and transition probabilities.
    covariance_type (COVAR_TYPES):
        Type of covariance parameters to use for the emission distributions.
    min_covar (float):
        Floor value for covariance matrices.
    random_state (Optional[int]):
        Random seed to use for reproducible results.
    verbose (bool):
        Whether to print progress logs during fitting.
    """

    COVAR_TYPES = Literal['spherical', 'tied', 'diag', 'full']

    def __init__(self,
                 n_states: int,
                 n_features: int,
                 max_duration: int,
                 params_init: bool = False,
                 k_means: bool = False,
                 alpha: float = 1.0,
                 min_covar: float = 1e-3,
                 covariance_type: COVAR_TYPES = 'full',
                 random_state: Optional[int] = None,
                 device: Optional[torch.device] = None):

        BaseHSMM.__init__(self,n_states,max_duration,params_init,alpha,random_state,device)
        
        GaussianEmissions.__init__(self,n_states,n_features,params_init,k_means,covariance_type,min_covar,random_state,device)
                    
    def __str__(self):
        return f'GaussianHSMM(n_states={self.n_states}, n_durations={self.max_duration},n_features={self.n_features})'

    @property
    def n_fit_params(self):
        """Return the number of trainable model parameters."""
        return {
            'states': self.n_states,
            'transitions': self.n_states**2,
            'durations': self.n_states * self.max_duration,
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
        return {'pi':self.initial_vector.matrix,
                'A':self.transition_matrix.matrix,
                'D':self.duration_matrix.matrix,
                'means': self.means, 
                'covars': self.covs}

    @property
    def dof(self):
        return self.n_states**2 - 1 + self.means.numel() + self.covs.numel() 
    
    def check_sequence(self, sequence, lengths=None):
        return validate_sequence(sequence,False,self.n_features,lengths)
    
    def map_emission(self, emission):
        return GaussianEmissions.map_emission(self,emission)

    def sample_B_params(self,X=None,seed=None):
        self._means,self._covs = GaussianEmissions.sample_emissions_params(self,self.min_covar,X,seed)

    def update_B_params(self, X, log_gamma, theta=None):
        gamma = [gamma.exp() for gamma in log_gamma]
        GaussianEmissions._update_emissions_params(self,X,gamma,theta)


class GaussianMixtureHSMM(BaseHSMM, GaussianMixtureEmissions):
    """
    Gaussian Hidden Semi-Markov Model (Gaussian HSMM)
    ----------
    This model assumes that the data follow a multivariate Gaussian distribution. 
    The model parameters (initial state probabilities, transition probabilities, duration probabilities,emission means, and emission covariances) 
    are learned using the Baum-Welch algorithm.

    Parameters:
    ----------
    n_states (int):
        Number of hidden states in the model.
    max_duration (int):
        Maximum duration of the states.
    n_features (int):
        Number of features in the emission data.
    n_components (int):
        Number of components in the Gaussian mixture model.
    params_init (bool):
        Whether to initialize the model parameters prior to fitting.
    alpha (float):
        Dirichlet concentration parameter for the prior over initial state probabilities and transition probabilities.
    covariance_type (COVAR_TYPES):
        Type of covariance parameters to use for the emission distributions.
    min_covar (float):
        Floor value for covariance matrices.
    random_state (Optional[int]):
        Random seed to use for reproducible results.
    verbose (bool):
        Whether to print progress logs during fitting.
    """

    COVAR_TYPES = Literal['spherical', 'tied', 'diag', 'full']

    def __init__(self,
                 n_states: int,
                 n_features: int,
                 max_duration: int,
                 n_components: int = 1,
                 params_init: bool = False,
                 k_means: bool = False,
                 alpha: float = 1.0,
                 covariance_type: COVAR_TYPES = 'full',
                 min_covar: float = 1e-3,
                 random_state: Optional[int] = None,
                 device: Optional[torch.device] = None):

        BaseHSMM.__init__(self,n_states,max_duration,params_init,alpha,random_state,device)
        
        GaussianMixtureEmissions.__init__(self,n_states,n_components,n_features,params_init,k_means,alpha,covariance_type,min_covar,random_state,device)
                    
    def __str__(self):
        return f'GaussianMixtureHSMM(n_states={self.n_states}, n_durations={self.max_duration},n_features={self.n_features}, n_components={self.n_components})'

    @property
    def n_fit_params(self):
        """Return the number of trainable model parameters."""
        fit_params_dict = {
            'states': self.n_states,
            'transitions': self.n_states**2,
            'durations': self.n_states * self.max_duration,
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
        return {'pi':self.initial_vector.matrix,
                'A':self.transition_matrix.matrix,
                'D':self.duration_matrix.matrix,
                'weights': self.weights.matrix,
                'means': self.means, 
                'covars': self.covs}

    @property
    def dof(self):
        return self.n_states**2 - 1 + self.n_states*self.n_components - self.n_states + self.means.numel() + self.covs.numel() 
    
    def check_sequence(self, sequence, lengths=None):
        return validate_sequence(sequence,False,self.n_features,lengths)
    
    def map_emission(self, emission):
        return GaussianMixtureEmissions.map_emission(self,emission)

    def sample_B_params(self,X,seed=None):
        self._means, self._covs = GaussianMixtureEmissions.sample_emissions_params(self,self.min_covar,X,seed)

    def update_B_params(self, X, log_gamma, theta=None):
        posterior_vec = []
        resp_vec = GaussianMixtureEmissions._compute_responsibilities(self,X)
        for resp,gamma_val in zip(resp_vec,log_gamma):
            posterior_vec.append(torch.exp(resp + gamma_val.T.unsqueeze(1)))

        GaussianMixtureEmissions._update_params(self,X,posterior_vec,theta)