from typing import Optional, List
import torch
import torch.nn as nn
from torch.distributions import Poisson, Independent

from chadhmm.hsmm.BaseHSMM import BaseHSMM
from chadhmm.utilities import utils

class PoissonHSMM(BaseHSMM):
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
        
        BaseHSMM.__init__(self,n_states,n_features,max_duration,alpha,seed)

    @property
    def dof(self):
        return self.n_states ** 2 + self.n_states * self.n_features - self.n_states - 1 + self.params.rates.numel()
    
    @property
    def pdf(self) -> Independent:
        return Independent(Poisson(self.params.rates),1)
    
    def sample_emission_params(self,X=None):
        if X is not None:
            rates = X.mean(dim=0).expand(self.n_states,-1).clone()
        else:
            rates = torch.ones(size=(self.n_states, self.n_features), 
                               dtype=torch.float64)

        return nn.ParameterDict({
            'rates':nn.Parameter(rates,requires_grad=False)
        })

    def estimate_emission_params(self,X,posterior,theta):
        return nn.ParameterDict({
            'rates':nn.Parameter(self._compute_rates(X,posterior,theta),requires_grad=False)
        })

    def _compute_rates(self,
                       X:List[torch.Tensor],
                       posterior:List[torch.Tensor],
                       theta:Optional[utils.ContextualVariables]) -> torch.Tensor:
        """Compute the rates for each hidden state"""
        new_rates = torch.zeros(size=(self.n_states, self.n_features), 
                               dtype=torch.float64)
        
        denom = torch.zeros(size=(self.n_states,1), 
                            dtype=torch.float64)
        
        for seq,gamma_val in zip(X,posterior):
            if theta is not None:
                # TODO: matmul shapes are inconsistent 
                raise NotImplementedError('Contextualized emissions not implemented for PoissonHMM')
            else:
                new_rates += gamma_val.T @ seq.double()
                denom += gamma_val.T.sum(dim=-1,keepdim=True)

        return new_rates / denom