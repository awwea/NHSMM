from typing import Optional
import torch

from .BaseHMM import BaseHMM # type: ignore
from ..emissions import PoissonEmissions # type: ignore


class PoissonHMM(BaseHMM, PoissonEmissions):
    """
    Poisson Hidden Markov Model (HMM)
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
    device (torch.device):
        Device on which to fit the model.
    """

    def __init__(self,
                 n_states: int,
                 n_features: int,
                 alpha: float = 1.0,
                 seed: Optional[int] = None, 
                 device: Optional[torch.device] = None):
        
        BaseHMM.__init__(self,n_states,alpha,seed,device)
        
        PoissonEmissions.__init__(self,n_states,n_features,device)

    @property
    def params(self):
        return {
            'pi': self.initial_vector.logits,
            'A': self.transition_matrix.logits,
            'rates': self.lambdas
        }

    @property
    def n_fit_params(self):
        return {
            'initial_states': self.n_states,
            'transitions': self.n_states**2,
            'rates': self.n_states * self.n_features   
        }

    @property
    def dof(self):
        return self.n_states ** 2 + self.n_states * self.n_features - self.n_states - 1

    def _update_B_params(self,X,log_gamma,theta):
        gamma = [torch.exp(gamma) for gamma in log_gamma]
        PoissonEmissions.update_emission_params(self,X,gamma,theta)

    def check_sequence(self,X):
        return PoissonEmissions.check_constraints(self,X)

    def map_emission(self,x):
        return PoissonEmissions.map_emission(self,x)

    def sample_B_params(self,X=None):
        self._lambdas = PoissonEmissions.sample_emission_params(self,X)