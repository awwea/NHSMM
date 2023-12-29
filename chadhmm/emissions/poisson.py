from typing import Optional, List
import torch
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
    device (torch.device):
        Device to use for computations.
    """

    def __init__(self, 
                 n_dims:int,
                 n_features:int,
                 device:Optional[torch.device] = None):
        
        BaseEmission.__init__(self,n_dims,n_features,device)

        self._lambdas = self.sample_emission_params()

    @property
    def lambdas(self) -> torch.Tensor:
        return self._lambdas
    
    @lambdas.setter
    def lambdas(self, lambdas: torch.Tensor):
        target_size = (self.n_dims, self.n_features)
        assert lambdas.shape == target_size, ValueError(f'lambdas shape must be {target_size}, got {lambdas.shape}')
        self._lambdas = lambdas.to(self.device)

    @property
    def pdf(self) -> Independent:
        return Independent(Poisson(self.lambdas),1)
    
    def map_emission(self,x):
        b_size = (-1,self.n_dims,-1) if x.ndim == 2 else (self.n_dims,-1)
        x_batched = x.unsqueeze(-2).expand(b_size)
        return self.pdf.log_prob(x_batched)
    
    def sample_emission_params(self,X=None):
        if X is not None:
            means = X.mean(dim=0).expand(self.n_dims,-1).clone()
        else:
            means = torch.ones(size=(self.n_dims, self.n_features), 
                                dtype=torch.float64, 
                                device=self.device) 

        return means
    
    def update_emission_params(self,X,posterior,theta=None):
        self._lambdas.copy_(self._compute_rates(X,posterior,theta))

    def _compute_rates(self,
                       X:List[torch.Tensor],
                       posterior:List[torch.Tensor],
                       theta:Optional[ContextualVariables]) -> torch.Tensor:
        """Compute the means for each hidden state"""
        new_rates = torch.zeros(size=(self.n_dims, self.n_features), 
                               dtype=torch.float64, 
                               device=self.device)
        
        denom = torch.zeros(size=(self.n_dims,1), 
                            dtype=torch.float64, 
                            device=self.device)
        
        for seq,gamma_val in zip(X,posterior):
            if theta is not None:
                # TODO: matmul shapes are inconsistent 
                raise NotImplementedError('Contextualized emissions not implemented for GaussianEmissions')
            else:
                new_rates += gamma_val.T @ seq
                denom += gamma_val.T.sum(dim=-1,keepdim=True)

        return new_rates / denom