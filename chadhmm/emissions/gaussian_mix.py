import torch
from torch.distributions import MultivariateNormal # type: ignore
from typing import Optional, Literal, Tuple, List
from sklearn.cluster import KMeans # type: ignore

from .base_mixture import MixtureEmissions # type: ignore
from ..stochastic_matrix import WeightsMatrix # type: ignore
from ..utils import ContextualVariables, validate_means, validate_covars, fill_covars # type: ignore
    

class GaussianMixtureEmissions(MixtureEmissions):
    """
    Gaussian Mixture Model for HMM emissions.    
    
    Parameters:
    ----------
    n_dims (int):
        Number of mixtures in the model. This is equal to the number of hidden states in the HMM.
    n_components (int):
        Number of components in the mixture model.
    n_features (int):
        Number of features in the data.
    init_weights (bool):
        Whether to initialize the mixture weights prior to fitting.
    init_dist ():
        Distribution to use for initializing the mixture weights.
    alpha (float):
        Dirichlet concentration parameter for the prior over mixture weights.
    k_means (bool):
        Whether to initialize the mixture means using K-Means clustering.
    min_covar (float):
        Minimum covariance for the mixture components.
    covariance_type (COVAR_TYPES):
        Type of covariance matrix to use for the mixture components. One of 'spherical', 'tied', 'diag', 'full'.
    dims (Sequence[str]):
        Names of the mixtures in the model.
    seed (int):
        Random seed for reproducibility.
    device (torch.device):
        Device to use for computations.
    """

    COVAR_TYPES_HINT = Literal['spherical', 'tied', 'diag', 'full']

    def __init__(self, 
                 n_dims: int,
                 n_features: int,
                 n_components: int,
                 params_init: bool = True,
                 k_means: bool = False,
                 alpha: float = 1.0,
                 covariance_type: COVAR_TYPES_HINT = 'full',
                 min_covar: float = 1e-3,
                 seed: Optional[int] = None,
                 device: Optional[torch.device] = None):
        
        MixtureEmissions.__init__(self,n_dims,n_features,n_components,alpha,params_init,seed,device)
        
        self.k_means = k_means
        self.min_covar = min_covar
        self.covariance_type = covariance_type
        if params_init:
            self._weights, self._means, self._covs = self.sample_emissions_params(seed=seed)
            
    @property
    def means(self) -> torch.Tensor:
        return self._means
    
    @means.setter
    def means(self, means: torch.Tensor):
        valid_means = validate_means(means, self.n_dims, self.n_features, self.n_components)
        self._means = valid_means.to(self.device)

    @property
    def covs(self) -> torch.Tensor:
        return self._covs

    @covs.setter
    def covs(self, new_covars: torch.Tensor):
        """Setter function for the covariance matrices."""
        valid_covars = validate_covars(new_covars, self.covariance_type, self.n_dims, self.n_features, self.n_components)
        self._covs = fill_covars(valid_covars, self.covariance_type, self.n_dims, self.n_features, self.n_components).to(self.device)

    @property
    def pdf(self) -> MultivariateNormal:
        return MultivariateNormal(self.means,self.covs)
    
    @property
    def params(self):
        return {'weights': self.weights.matrix,
                'means': self.means, 
                'covs': self.covs}
    
    def sample_emissions_params(self,X=None,seed=None) -> Tuple[WeightsMatrix,torch.Tensor,torch.Tensor]:
        """Initialize the emission parameters."""
        new_weights = self.sample_weights(seed)
        if X is not None:
            means = self._sample_kmeans(X,seed) if self.k_means else X.mean(dim=0,keepdim=True).expand(self.n_dims,self.n_components,-1).clone()
            centered_data = X - X.mean(dim=0)
            covs = (torch.mm(centered_data.T, centered_data) / (X.shape[0] - 1)).expand(self.n_dims,self.n_components,-1,-1).clone()
        else:
            means = torch.zeros(size=(self.n_dims, self.n_components, self.n_features), 
                                dtype=torch.float64, 
                                device=self.device)
        
            covs = self.min_covar + torch.eye(n=self.n_features, 
                                              dtype=torch.float64,
                                              device=self.device).expand((self.n_dims, self.n_components, self.n_features, self.n_features)).clone()

        return new_weights, means, covs

    def update_emission_params(self,X,resp,theta=None):
        self._weights.matrix.copy_(self._compute_weights(resp))
        self._means.copy_(self._compute_means(X,resp,theta)) 
        self._covs.copy_(self._compute_covs(X,resp,theta))
    
    def _sample_kmeans(self,
                       X:torch.Tensor,
                       seed:Optional[int]=None) -> torch.Tensor:
        """Sample cluster means from K Means algorithm"""
        k_means_alg = KMeans(n_clusters=self.n_dims, 
                               random_state=seed, 
                               n_init="auto").fit(X)
        return torch.from_numpy(k_means_alg.cluster_centers_).reshape(self.n_dims,self.n_components,self.n_features)
    
    def _compute_means(self,
                       X:List[torch.Tensor], 
                       responsibilities:List[torch.Tensor],
                       theta:Optional[ContextualVariables]=None) -> torch.Tensor:
        """Compute the means for each hidden state"""
        new_mean = torch.zeros(size=(self.n_dims,self.n_components,self.n_features), 
                               dtype=torch.float64, 
                               device=self.device)
        
        denom = torch.zeros(size=(self.n_dims,self.n_components,1), 
                            dtype=torch.float64, 
                            device=self.device)
        
        for seq,resp in zip(X,responsibilities):
            if theta is not None:
                # TODO: matmul shapes are inconsistent
                raise NotImplementedError('Contextual variables are not yet implemented')
            else:
                new_mean += resp @ seq
                denom += resp.sum(dim=-1,keepdim=True)

        return new_mean / denom
    
    def _compute_covs(self,
                      X:List[torch.Tensor],
                      responsibilities:List[torch.Tensor],
                      theta:Optional[ContextualVariables]=None) -> torch.Tensor:
        """Compute the covariances for each component."""
        new_covs = torch.zeros(size=(self.n_dims,self.n_components,self.n_features,self.n_features), 
                               dtype=torch.float64, 
                               device=self.device)
        
        denom = torch.zeros(size=(self.n_dims,self.n_components,1,1), 
                            dtype=torch.float64, 
                            device=self.device)

        for seq,resp in zip(X,responsibilities):
            if theta is not None:
                # TODO: matmul shapes are inconsistent 
                raise NotImplementedError('Contextualized emissions not implemented for GaussianEmissions')
            else:
                resp_expanded = resp.unsqueeze(-1)
                diff = seq.unsqueeze(0).expand(self.n_dims,self.n_components,-1,-1) - self.means.unsqueeze(2)
                new_covs += torch.transpose(diff * resp_expanded,2,3) @ diff
                denom += torch.sum(resp_expanded,dim=-2,keepdim=True)

        new_covs /= denom
        new_covs += self.min_covar * torch.eye(self.n_features, device=self.device)
        
        return new_covs
    