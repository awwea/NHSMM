from typing import Optional
import torch
from torch.distributions import Multinomial

from chadhmm.hsmm.BaseHSMM import BaseHSMM
from chadhmm.utilities import utils, constraints


class MultinomialHSMM(BaseHSMM):
    """
    Categorical Hidden semi-Markov Model (HSMM)
    ----------
    Hidden semi-Markov model with categorical (discrete) emissions. This model is an extension of classical HMMs where the duration of each state is modeled by a geometric distribution.
    Duration in each state is modeled by a Categorical distribution with a fixed maximum duration.

    Parameters:
    ----------
    n_states (int):
        Number of hidden states in the model.
    n_features (int): 
        Number of emissions in the model.
    seed (int):
        Random seed for reproducibility.
    max_duration (int):
        Maximum duration of each state.
    params_init (bool):
        Whether to initialize the model parameters prior to fitting.
    init_dist (SAMPLING_DISTRIBUTIONS):
        Distribution to use for initializing the model parameters.
    alpha (float):
        Dirichlet concentration parameter for the prior over initial state probabilities and transition probabilities.
    verbose (bool):
        Whether to print progress logs during fitting.
    """
    def __init__(self,
                 n_states:int,
                 n_features:int,
                 max_duration:int,
                 n_trials:int = 1,
                 alpha:float = 1.0,
                 seed:Optional[int] = None):
        
        self.n_features = n_features
        self.n_trials = n_trials
        super().__init__(n_states,max_duration,alpha,seed)

    @property
    def dof(self):
        return self.n_states ** 2 + self.n_states * self.n_features - self.n_states - 1

    def sample_emission_pdf(self,X=None):
        if X is not None:
            emission_freqs = torch.bincount(X) / X.shape[0]
            emission_matrix = torch.log(emission_freqs.expand(self.n_states,-1))
        else:
            emission_matrix = torch.log(constraints.sample_probs(self.alpha,(self.n_states,self.n_features)))

        return Multinomial(total_count=self.n_trials,logits=emission_matrix)

    def _estimate_emission_pdf(self,X,posterior,theta=None):
        new_B = self._compute_B(X,posterior,theta)
        return Multinomial(total_count=self.n_trials,logits=new_B)

    def _compute_B(self,
                   X:torch.Tensor,
                   posterior:torch.Tensor,
                   theta:Optional[utils.ContextualVariables]=None) -> torch.Tensor: 
        """Compute the emission probabilities for each hidden state."""
        if theta is not None:
            #TODO: Implement contextualized emissions
            raise NotImplementedError('Contextualized emissions not implemented for CategoricalEmissions')
        else:  
            new_B = posterior @ X
            new_B /= posterior.sum(1,keepdim=True)

        return new_B