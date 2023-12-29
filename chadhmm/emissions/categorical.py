import torch
from torch.distributions import Categorical # type: ignore
from typing import Optional, List

from .base_emiss import BaseEmission # type: ignore
from ..stochastic_matrix import StochasticTensor, MAT_OPS # type: ignore
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
    alpha (float):
        Dirichlet concentration parameter for the prior over emission probabilities.
    device (torch.device):
        Device on which to fit the model.
    """

    def __init__(self,
                 n_dims:int,
                 n_features:int,
                 alpha:float = 1.0,
                 device:Optional[torch.device] = None):
        
        BaseEmission.__init__(self,n_dims,n_features,device)

        self.alpha = alpha
        self._emission_matrix = self.sample_emission_params()

    @property
    def pdf(self) -> Categorical:
        return self.emission_matrix.pmf

    @property
    def emission_matrix(self) -> StochasticTensor:
        return self._emission_matrix
    
    @emission_matrix.setter
    def emission_matrix(self, matrix):
        self.emission_matrix.logits = matrix

    def sample_emission_params(self,X=None) -> StochasticTensor:
        if X is not None:
            emission_freqs = torch.bincount(X) / X.shape[0]
            return StochasticTensor(name='Emission', 
                                    logits=torch.log(emission_freqs.expand(self.n_dims,-1)),
                                    device=self.device)
        else:
            return StochasticTensor.from_dirichlet(name='Emission',
                                                   size=(self.n_dims,self.n_features),
                                                   prior=self.alpha,
                                                   device=self.device)

    def map_emission(self,x):
        batch_shaped = x.repeat(self.n_dims,1).T
        return self.pdf.log_prob(batch_shaped)

    def update_emission_params(self,X,posterior,theta=None):
        self._emission_matrix._logits.copy_(self._compute_emprobs(X,posterior,theta))

    def _compute_emprobs(self,
                        X:List[torch.Tensor],
                        posterior:List[torch.Tensor],
                        theta:Optional[ContextualVariables]) -> torch.Tensor: 
        """Compute the emission probabilities for each hidden state."""
        emission_mat = torch.zeros(size=(self.n_dims, self.n_features),
                                   dtype=torch.float64,
                                   device=self.device)

        for seq,gamma_val in zip(X,posterior):
            if theta is not None:
                #TODO: Implement contextualized emissions
                raise NotImplementedError('Contextualized emissions not implemented for CategoricalEmissions')
            else:
                masks = seq.view(1,-1) == self.pdf.enumerate_support(expand=False)
                for i,mask in enumerate(masks):
                    masked_gamma = gamma_val[mask]
                    emission_mat[:,i] += masked_gamma.sum(dim=0)

        return log_normalize(emission_mat.log(),1)