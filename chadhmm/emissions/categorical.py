import torch
from torch.distributions import Categorical # type: ignore
from typing import Optional, List

from .base_emiss import BaseEmission # type: ignore
from ..stochastic_matrix import EmissionMatrix # type: ignore
from ..utils import ContextualVariables, log_normalize # type: ignore


class CategoricalEmissions(BaseEmission):
    """
    Categorical emission distribution for HMMs.

    Parameters:
    ----------
    n_dims (int):
        Number of hidden states in the model.
    n_features (int):
        Number of emissions in the model.
    init_params (bool):
        Whether to initialize the emission parameters.
    alpha (float):
        Dirichlet concentration parameter for the prior over emission probabilities.
    seed (int):
        Random seed for reproducibility.
    device (torch.device):
        Device on which to fit the model.

    Attributes:
    ----------
    emission_matrix (EmissionMatrix):
        Emission matrix representing the categorical distribution.
    """

    def __init__(self,
                 n_dims:int,
                 n_features:int,
                 init_params:bool=True,
                 alpha:float = 1.0,
                 seed:Optional[int] = None,
                 device:Optional[torch.device] = None):
        
        super().__init__(n_dims,n_features,True,seed,device)

        self.alpha = alpha
        if init_params:
            self._emission_matrix = self.sample_emissions_params(seed=seed)

    def __str__(self):
        return f'CategoricalEmissions(n_dims={self.n_dims}, n_features={self.n_features})'

    @property
    def pdf(self) -> Categorical:
        """Compute the probability density of the emission distribution."""
        return self._emission_matrix._dist

    @property
    def emission_matrix(self) -> EmissionMatrix:
        return self._emission_matrix
    
    @emission_matrix.setter
    def emission_matrix(self, matrix:torch.Tensor):
        assert (self.n_dims,self.n_features) == matrix.shape, f'Expected matrix shape {(self.n_dims,self.n_features)} but got {matrix.shape}'
        if isinstance(matrix, EmissionMatrix):
            self._emission_matrix = matrix
        elif isinstance(matrix, torch.Tensor):
            self._emission_matrix = EmissionMatrix(n_states=self.n_dims, 
                                                   n_emissions=self.n_features, 
                                                   matrix=matrix,
                                                   device=self.device)
        else:
            raise NotImplementedError('Matrix type not supported')

    def sample_emissions_params(self,X=None,seed=None):
        if X is not None:
            emission_freqs = torch.bincount(X) / X.shape[0]
            return EmissionMatrix.from_tensor(emission_freqs.expand(self.n_dims,-1).log())
        else:
            return EmissionMatrix(n_states=self.n_dims,
                                  n_emissions=self.n_features,
                                  rand_seed=seed,
                                  alpha=self.alpha,
                                  device=self.device)

    def map_emission(self,x):
        return self.pdf.log_prob(x.repeat(self.n_dims,1).T)

    def update_emission_params(self,X,gamma,theta=None):
        self._emission_matrix.matrix.copy_(self._compute_emprobs(X,gamma,theta))

    def _compute_emprobs(self, 
                        X:List[torch.Tensor],
                        gamma:List[torch.Tensor],
                        theta:Optional[ContextualVariables]) -> torch.Tensor:  
        """Compute the emission probabilities for each hidden state."""
        emission_mat = torch.zeros(size=(self.n_dims, self.n_features),
                                   dtype=torch.float64,
                                   device=self.device)

        for seq,gamma_val in zip(X,gamma):
            if theta is not None:
                #TODO: Implement contextualized emissions
                raise NotImplementedError('Contextualized emissions not implemented for CategoricalEmissions')
            else:
                masks = seq.view(1,-1) == torch.arange(self.n_features).view(-1,1)
                for i,mask in enumerate(masks):
                    masked_gamma = gamma_val[mask]
                    emission_mat[:,i] += masked_gamma.sum(dim=0)

        return log_normalize(emission_mat.log(),1)