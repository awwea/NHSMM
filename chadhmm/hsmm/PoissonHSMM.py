from typing import Optional
import torch

from .BaseHSMM import BaseHSMM # type: ignore
from ..emissions import PoissonEmissions # type: ignore


class PoissonHMM(BaseHSMM):
    """
    Poisson Hidden Semi-Markov Model (HSMM)
    ----------
    Hidden Markov model with emissions. This model is a special case of the HSMM model with a geometric duration distribution.

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
                 n_states: int,
                 n_features: int,
                 max_duration:int,
                 alpha: float = 1.0,
                 seed: Optional[int] = None):
        
        BaseHSMM.__init__(self,n_states,max_duration,alpha,seed)
        self.emissions = PoissonEmissions(n_states,n_features)

    @property
    def n_fit_params(self):
        return {
            'initial_states': self.n_states,
            'transitions': self.n_states**2,
            'rates': self.n_states * self.emissions.n_features   
        }

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
        self._lambdas = self.emissions.sample_emission_params(X)