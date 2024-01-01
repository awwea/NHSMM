from typing import Optional
import torch

from .BaseHMM import BaseHMM # type: ignore
from ..emissions import CategoricalEmissions # type: ignore


class CategoricalHMM(BaseHMM):
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
    alpha (float):
        Dirichlet concentration parameter for the prior over initial distribution, transition amd emission probabilities.
    seed (int):
        Random seed for reproducibility.
    """

    def __init__(self,
                 n_states:int,
                 n_features:int,
                 alpha:float = 1.0,
                 seed:Optional[int] = None):
        
        BaseHMM.__init__(self,n_states,alpha,seed)
        self.add_module('emissions',
                        CategoricalEmissions(n_states,n_features,alpha))

    @property
    def dof(self):
        return self.n_states ** 2 + self.n_states * self.emissions.n_features - self.n_states - 1

    def _update_B_params(self,X,log_gamma,theta):
        gamma = [torch.exp(gamma) for gamma in log_gamma]
        self.emissions.update_emission_params(X,gamma,theta)

    def check_sequence(self,X):
        return self.emissions.check_constraints(X)

    def map_emission(self,x):
        return self.emissions.map_emission(x)

    def sample_B_params(self,X=None):
        self.emissions.emission_matrix = self.emissions.sample_emission_params(X)


