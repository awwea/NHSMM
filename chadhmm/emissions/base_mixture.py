import torch
from abc import ABC
from typing import List
from torch.distributions import MixtureSameFamily

from ..emissions.base_emiss import BaseEmission
from ..stochastic_matrix import StochasticTensor
from ..utils import log_normalize


class MixtureEmissions(BaseEmission, ABC):
    """
    Mixture model for HMM emissions. This class is an abstract base class for Gaussian, Poisson and other mixture models.
    """

    def __init__(self, 
                 n_dims:int,
                 n_features:int,
                 n_components:int,
                 alpha:float = 1.0):

        BaseEmission.__init__(self,n_dims,n_features)

        self.n_components = n_components
        self.alpha = alpha
        self.weights = self.sample_weights(alpha)

    def __str__(self):
        return BaseEmission.__str__(self).replace(')',f', n_components={self.n_components})')
    
    def extra_repr(self) -> str:
        return super().extra_repr() + f'\nn_components = {self.n_components}'
        
    @property
    def mixture_pdf(self) -> MixtureSameFamily:
        """Return the emission distribution for Gaussian Mixture Distribution."""
        return MixtureSameFamily(mixture_distribution = self.weights.pmf,
                                 component_distribution = self.pdf)
    
    def sample_weights(self, alpha:float = 1.0) -> StochasticTensor:
        """Sample the weights for the mixture."""
        return StochasticTensor.from_dirichlet(name='Weights',
                                               size=(self.n_dims, self.n_components),
                                               prior=alpha)    

    def map_emission(self, x:torch.Tensor) -> torch.Tensor:
        b_size = (-1,self.n_dims,-1) if x.ndim == 2 else (self.n_dims,-1)
        x_batched = x.unsqueeze(-2).expand(b_size)
        return self.mixture_pdf.log_prob(x_batched)
    
    def _compute_responsibilities(self, X:List[torch.Tensor]) -> List[torch.Tensor]:
        """Compute the responsibilities for each component."""
        resp_vec = []
        for seq in X:
            n_observations = seq.size(dim=0)
            log_responsibilities = torch.zeros(size=(self.n_dims,self.n_components,n_observations), 
                                               dtype=torch.float64)

            for t in range(n_observations):
                log_responsibilities[...,t] = log_normalize(self.weights.logits + self.pdf.log_prob(seq[t]),1)

            resp_vec.append(log_responsibilities)
        
        return resp_vec
    
    def _compute_weights(self, posterior:List[torch.Tensor]) -> torch.Tensor:
        log_weights = torch.zeros(size=(self.n_dims,self.n_components),
                                  dtype=torch.float64)

        for p in posterior:
            log_weights += p.exp().sum(-1)
        
        return log_normalize(log_weights.log(),1)