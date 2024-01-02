from typing import Optional, List
import torch
import torch.nn as nn
from torch.distributions import Categorical # type: ignore

from .BaseHMM import BaseHMM # type: ignore
from ..utils import ContextualVariables, log_normalize, sample_logits # type: ignore


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
        
        super().__init__(n_states,n_features,alpha,seed)

    @property
    def dof(self):
        return self.n_states ** 2 + self.n_states * self.n_features - self.n_states - 1
    
    @property
    def pdf(self) -> Categorical:
        return Categorical(logits=self.params.B)

    def estimate_emission_params(self,X,posterior,theta=None):
        return nn.ParameterDict({
            'B':nn.Parameter(
                self._compute_emprobs(X,posterior,theta),
                requires_grad=False
            )
        })

    def sample_emission_params(self,X=None):
        if X is not None:
            emission_freqs = torch.bincount(X) / X.shape[0]
            emission_matrix = torch.log(emission_freqs.expand(self.n_states,-1))
        else:
            emission_matrix = sample_logits(self.alpha,(self.n_states,self.n_features),False)
            
        return nn.ParameterDict({
            'B':nn.Parameter(
                emission_matrix,
                requires_grad=False
            )
        })

    def _compute_emprobs(self,
                        X:List[torch.Tensor],
                        posterior:List[torch.Tensor],
                        theta:Optional[ContextualVariables]=None) -> torch.Tensor: 
        """Compute the emission probabilities for each hidden state."""
        emission_mat = torch.zeros(size=(self.n_states,self.n_features),
                                   dtype=torch.float64)

        for seq,gamma_val in zip(X,posterior):
            if theta is not None:
                #TODO: Implement contextualized emissions
                raise NotImplementedError('Contextualized emissions not implemented for CategoricalEmissions')
            else:
                masks = seq.view(1,-1) == self.pdf.enumerate_support(expand=False)
                for i,mask in enumerate(masks):
                    emission_mat[:,i] += gamma_val[mask].sum(dim=0)

        return log_normalize(emission_mat.log(),1)


