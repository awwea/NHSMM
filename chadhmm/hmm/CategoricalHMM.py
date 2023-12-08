from typing import Optional
import torch

from .BaseHMM import BaseHMM # type: ignore
from ..emissions import CategoricalEmissions # type: ignore
from ..utils import validate_sequence # type: ignore

class CategoricalHMM(BaseHMM, CategoricalEmissions):
    """
    Categorical Hidden Markov Model (HMM)
    ----------
    Hidden Markov model with categorical (discrete) emissions. This model is a special case of the HSMM model with a geometric duration distribution.

    Parameters:
    ----------
    n_states (int):
        Number of hidden states in the model.
    n_features (int): 
        Number of emissions in the model.
    random_state (int):
        Random seed for reproducibility.
    params_init (bool):
        Whether to initialize the model parameters.
    alpha (float):
        Dirichlet concentration parameter for the prior over initial distribution, transition amd emission probabilities.
    verbose (bool):
        Whether to print progress logs during fitting.
    device (torch.device):
        Device on which to fit the model.
    """

    def __init__(self,
                 n_states: int,
                 n_features: int,
                 alpha: float = 1.0,
                 params_init: bool = False,
                 seed: Optional[int] = None, 
                 device: Optional[torch.device] = None):
        
        BaseHMM.__init__(self,n_states,params_init,alpha,seed,device)
        
        CategoricalEmissions.__init__(self,n_states,n_features,params_init,alpha,seed,device)
        
    def __str__(self):
        return f'CategoricalHMM(n_states={self.n_states}, n_feautures={self.n_features})'

    @property
    def params(self):
        return {
            'pi': self.initial_vector.matrix,
            'A': self.transition_matrix.matrix,
            'B': self.emission_matrix.matrix
        }

    @property
    def n_fit_params(self):
        return {
            'initial_states': self.n_states,
            'transitions': self.n_states**2,
            'emissions': self.n_states * self.n_features
        }

    @property
    def dof(self):
        return self.n_states ** 2 + self.n_states * self.n_features - self.n_states - 1

    def _update_B_params(self,X,log_gamma,theta):
        gamma = [torch.exp(gamma) for gamma in log_gamma]
        CategoricalEmissions.update_emission_params(self,X,gamma,theta)

    def check_sequence(self,sequence):
        return validate_sequence(sequence,True,self.n_features)

    def map_emission(self,x):
        return CategoricalEmissions.map_emission(self,x)

    def sample_B_params(self,X,seed=None):
        self._emission_matrix = CategoricalEmissions.sample_emissions_params(self,X,seed)


