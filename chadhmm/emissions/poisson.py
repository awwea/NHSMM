from typing import Optional, List
import torch
import torch.nn as nn
from torch.distributions import Poisson, Independent
from .base_emiss import BaseEmission
from ..utils import ContextualVariables


class PoissonEmissions(BaseEmission): 
    """
    (Multiple) Poisson Distribution for HMM emissions.    
    
    Parameters:
    ----------
    n_dims (int):
        Number of mixtures in the model. This is equal to the number of hidden states in the HMM.
    n_features (int):
        Number of features in the data. For m > 1 we assume emissions follow joint distribution composed of univariate marginals.
    """

    def __init__(self, 
                 n_dims:int,
                 n_features:int):
        
        BaseEmission.__init__(self,n_dims,n_features)
        self.lambdas:nn.Parameter = self.sample_emission_params().get('rates')

    @property
    def pdf(self) -> Independent:
        return Independent(Poisson(self.lambdas),1)
    
    def map_emission(self,x):
        b_size = (-1,self.n_dims,-1) if x.ndim == 2 else (self.n_dims,-1)
        x_batched = x.unsqueeze(-2).expand(b_size)
        return self.pdf.log_prob(x_batched)
    
    def sample_emission_params(self,X=None):
        if X is not None:
            rates = X.mean(dim=0).expand(self.n_dims,-1).clone()
        else:
            rates = torch.ones(size=(self.n_dims, self.n_features), 
                                dtype=torch.float64) 

        return nn.ParameterDict({'rates':nn.Parameter(rates,requires_grad=False)})
    
    def update_emission_params(self,X,posterior,theta=None):
        self.lambdas.data = self._compute_rates(X,posterior,theta)

    def _compute_rates(self,
                       X:List[torch.Tensor],
                       posterior:List[torch.Tensor],
                       theta:Optional[ContextualVariables]) -> torch.Tensor:
        """Compute the means for each hidden state"""
        new_rates = torch.zeros(size=(self.n_dims, self.n_features), 
                               dtype=torch.float64)
        
        denom = torch.zeros(size=(self.n_dims,1), 
                            dtype=torch.float64)
        
        for seq,gamma_val in zip(X,posterior):
            if theta is not None:
                # TODO: matmul shapes are inconsistent 
                raise NotImplementedError('Contextualized emissions not implemented for GaussianEmissions')
            else:
                new_rates += gamma_val.T @ seq.double()
                denom += gamma_val.T.sum(dim=-1,keepdim=True)

        return new_rates / denom