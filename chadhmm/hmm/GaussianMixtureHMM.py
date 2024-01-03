from typing import Optional, Literal, List
import torch
import torch.nn as nn
from torch.distributions import MultivariateNormal,MixtureSameFamily,Categorical
from sklearn.cluster import KMeans # type: ignore

from .BaseHMM import BaseHMM # type: ignore
from ..utils import ContextualVariables, log_normalize, sample_logits # type: ignore


class GaussianMixtureHMM(BaseHMM):
    """
    Gaussian Hidden Markov Model (Gaussian HMM)
    ----------
    This model assumes that the data follows a multivariate Gaussian distribution. 
    The model parameters (initial state probabilities, transition probabilities, emission means, and emission covariances) are learned using the Baum-Welch algorithm.

    Parameters:
    ----------
    n_states (int):
        Number of hidden states in the model.
    n_features (int):
        Number of features in the emission data.
    n_components (int):
        Number of components in the Gaussian mixture model.
    alpha (float):
        Dirichlet concentration parameter for the prior over initial state probabilities and transition probabilities.
    covariance_type (COVAR_TYPES):
        Type of covariance parameters to use for the emission distributions.
    min_covar (float):
        Floor value for covariance matrices.
    seed (Optional[int]):
        Random seed to use for reproducible results.
    """
    __slots__ = 'n_components'
    COVAR_TYPES = Literal['spherical', 'tied', 'diag', 'full']

    def __init__(self,
                 n_states:int,
                 n_features:int,
                 n_components:int = 1,
                 k_means:bool = False,
                 alpha:float = 1.0,
                 covariance_type:COVAR_TYPES = 'full',
                 min_covar:float = 1e-3,
                 seed:Optional[int] = None):

        self.min_covar = min_covar
        self.k_means = k_means
        self.covariance_type = covariance_type
        self.n_components = n_components
        BaseHMM.__init__(self,n_states,n_features,alpha,seed)
        
    @property
    def dof(self):
        return self.n_states**2 - 1 + self.n_states*self.n_components - self.n_states + self.params.means.numel() + self.params.covs.numel()
    
    @property
    def pdf(self) -> MixtureSameFamily:
        """Return the emission distribution for Gaussian Mixture Distribution."""
        return MixtureSameFamily(Categorical(logits=self.params.weights),
                                 MultivariateNormal(self.params.means,self.params.covs))
    
    def sample_emission_params(self,X=None):
        weights = sample_logits(self.alpha,(self.n_states,self.n_components),False)
        if X is not None:
            means = self._sample_kmeans(X) if self.k_means else X.mean(dim=0,keepdim=True).expand(self.n_states,self.n_components,-1).clone()
            centered_data = X - X.mean(dim=0)
            covs = (torch.mm(centered_data.T, centered_data) / (X.shape[0] - 1)).expand(self.n_states,self.n_components,-1,-1).clone()
        else:
            means = torch.zeros(size=(self.n_states, self.n_components, self.n_features), 
                                dtype=torch.float64)
        
            covs = self.min_covar + torch.eye(n=self.n_features, 
                                              dtype=torch.float64).expand((self.n_states, self.n_components, self.n_features, self.n_features)).clone()

        return nn.ParameterDict({
            'weights':nn.Parameter(weights,requires_grad=False),
            'means':nn.Parameter(means,requires_grad=False),
            'covs':nn.Parameter(covs,requires_grad=False)
        })
    
    def estimate_emission_params(self,X,posterior,theta):
        resp_vec = self._compute_responsibilities(X)
        posterior_vec = [torch.exp(resp + post.unsqueeze(-1)) for resp,post in zip(resp_vec,posterior)]

        return nn.ParameterDict({
            'weights':nn.Parameter(self._compute_weights(posterior_vec),requires_grad=False),
            'means':nn.Parameter(self._compute_means(X,posterior_vec,theta),requires_grad=False),
            'covs':nn.Parameter(self._compute_covs(X,posterior_vec,theta),requires_grad=False)
        })
    
    def _sample_kmeans(self, X:torch.Tensor, seed:Optional[int]=None) -> torch.Tensor:
        """Sample cluster means from K Means algorithm"""
        k_means_alg = KMeans(n_clusters=self.n_states, 
                             random_state=seed, 
                             n_init="auto").fit(X)
        return torch.from_numpy(k_means_alg.cluster_centers_).reshape(self.n_states,self.n_components,self.n_features)
    
    def _compute_responsibilities(self, X:List[torch.Tensor]) -> List[torch.Tensor]:
        """Compute the responsibilities for each component."""
        resp_vec = []
        for seq in X:
            n_observations = seq.size(dim=0)
            log_responsibilities = torch.zeros(size=(n_observations,self.n_states,self.n_components), 
                                               dtype=torch.float64)

            for t in range(n_observations):
                log_responsibilities[t] = log_normalize(self.params.weights + self.pdf.component_distribution.log_prob(seq[t]))

            resp_vec.append(log_responsibilities)
        
        return resp_vec
    
    def _compute_weights(self, posterior:List[torch.Tensor]) -> torch.Tensor:
        post_concat = torch.cat(posterior).sum(0).log()
        return log_normalize(post_concat)
    
    def _compute_means(self,
                       X:List[torch.Tensor], 
                       posterior:List[torch.Tensor],
                       theta:Optional[ContextualVariables]=None) -> torch.Tensor:
        """Compute the means for each hidden state"""
        new_mean = torch.zeros(size=(self.n_states,self.n_components,self.n_features), 
                               dtype=torch.float64)
        
        for seq,resp in zip(X,posterior):
            if theta is not None:
                # TODO: matmul shapes are inconsistent
                raise NotImplementedError('Contextualized emissions not implemented for GaussianMixtureHMM')
            else:
                new_mean += resp.permute(1,2,0) @ seq
        
        new_mean /= torch.cat(posterior).sum(0,keepdim=True).permute(1,2,0)
        return new_mean
    
    def _compute_covs(self,
                      X:List[torch.Tensor],
                      posterior:List[torch.Tensor],
                      theta:Optional[ContextualVariables]=None) -> torch.Tensor:
        """Compute the covariances for each component."""
        new_covs = torch.zeros(size=(self.n_states,self.n_components,self.n_features,self.n_features), 
                               dtype=torch.float64)
        
        for seq,resp in zip(X,posterior):
            if theta is not None:
                # TODO: matmul shapes are inconsistent 
                raise NotImplementedError('Contextualized emissions not implemented for GaussianMixtureHMM')
            else:
                resp_expanded = resp.permute(1,2,0).unsqueeze(-1)
                diff = seq.unsqueeze(0).expand(self.n_states,self.n_components,-1,-1) - self.params.means.unsqueeze(2)
                new_covs += torch.transpose(diff * resp_expanded,2,3) @ diff

        new_covs /= torch.cat(posterior).sum(0,keepdim=True).permute(1,2,0).unsqueeze(-1)
        new_covs += self.min_covar * torch.eye(self.n_features)
        
        return new_covs