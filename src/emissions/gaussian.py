from typing import Optional, Literal, Tuple, List
from sklearn.cluster import KMeans # type: ignore
import torch

from ..utils import ContextualVariables, validate_means, validate_covars, fill_covars # type: ignore


class GaussianEmissions:
    """
    Gaussian Distribution for HMM emissions.    
    
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
                 params_init: bool = True,
                 k_means: bool = False,
                 covariance_type: COVAR_TYPES_HINT = 'full',
                 min_covar: float = 1e-3,
                 seed: Optional[int] = None,
                 device: Optional[torch.device] = None):
        
        self.n_dims = n_dims
        self.n_features = n_features
        self.min_covar = min_covar
        self.covariance_type = covariance_type
        self.seed = seed
        self.k_means = k_means
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') if device is None else device

        if params_init:
            self._means, self._covs = self.sample_emissions_params()
            
    def __str__(self):
        return f'GaussianEmissions(n_dims={self.n_dims}, n_features={self.n_features})'        

    @property
    def means(self) -> torch.Tensor:
        return self._means
    
    @means.setter
    def means(self, means: torch.Tensor):
        valid_means = validate_means(means, self.n_dims, self.n_features)
        self._means = valid_means.to(self.device)

    @property
    def covs(self) -> torch.Tensor:
        return self._covs

    @covs.setter
    def covs(self, new_covars: torch.Tensor):
        """Setter function for the covariance matrices."""
        valid_covars = validate_covars(new_covars, self.covariance_type, self.n_dims, self.n_features)
        self._covs = fill_covars(valid_covars, self.covariance_type, self.n_dims, self.n_features).to(self.device)

    @property
    def pdf(self):
        return torch.distributions.MultivariateNormal(self.means,self.covs)
    
    def map_emission(self, x:torch.Tensor) -> torch.Tensor:
        """Sample the emission probabilities for each hidden state."""
        x_batched = x.unsqueeze(1).expand(-1,self.n_dims,-1)
        return self.pdf.log_prob(x_batched)
    
    @property
    def params(self):
        return {'means': self.means, 
                'covs': self.covs}
    
    def sample_emissions_params(self,
                                X:Optional[torch.Tensor]=None,
                                seed:Optional[int]=None) -> Tuple[torch.Tensor, torch.Tensor]:
        """Initialize the emission parameters."""
        if X is not None:
            means = self._sample_kmeans(X, seed) if self.k_means else X.mean(dim=0).expand(self.n_dims,-1).clone()
            centered_data = X - X.mean(dim=0)
            covs = (torch.mm(centered_data.T, centered_data) / (X.shape[0] - 1)).expand(self.n_dims,-1,-1).clone()
        else:
            means = torch.zeros(size=(self.n_dims, self.n_features), 
                                dtype=torch.float64, 
                                device=self.device) 
            covs = self.min_covar + torch.eye(n=self.n_features, 
                                              dtype=torch.float64,
                                              device=self.device).expand((self.n_dims, self.n_features, self.n_features)).clone()

        return means, covs
    
    def _sample_kmeans(self,
                       X:torch.Tensor,
                       seed:Optional[int]=None) -> torch.Tensor:
        """Sample cluster means from K Means algorithm"""
        k_means_alg = KMeans(n_clusters=self.n_dims, 
                               random_state=seed, 
                               n_init="auto").fit(X)
        return torch.from_numpy(k_means_alg.cluster_centers_).reshape(self.n_dims,self.n_features)

    def _compute_means(self,
                      X:List[torch.Tensor],
                      gamma:List[torch.Tensor],
                      theta:Optional[ContextualVariables]) -> torch.Tensor:
        """Compute the means for each hidden state"""
        new_mean = torch.zeros(size=(self.n_dims, self.n_features), 
                               dtype=torch.float64, 
                               device=self.device)
        
        denom = torch.zeros(size=(self.n_dims,1), 
                            dtype=torch.float64, 
                            device=self.device)
        
        for seq,gamma_val in zip(X,gamma):
            if theta is not None:
                # TODO: matmul shapes are inconsistent 
                raise NotImplementedError('Contextualized emissions not implemented for GaussianEmissions')
            else:
                new_mean += gamma_val.T @ seq
                denom += gamma_val.T.sum(dim=-1,keepdim=True)

        return new_mean / denom
    
    # def _contextual_matrix(self,
    #                        seq:torch.Tensor, 
    #                        gamma:torch.Tensor, 
    #                        theta:torch.Tensor) -> torch.Tensor:
    #     """Compute the contextualized means for each hidden state."""
    #     # resulting shape of nominator should be: (n_states, n_features, n_context)e
    #     # First shape is (n_dims, time, n_feautures) and the second shape is (n_dims, n_context, n_samples)
    #     # weighted_X = gamma.unsqueeze(dim=-1).expand(-1, -1, self.n_features) * seq
    #     # nominator = (weighted_X.transpose(-2,-1) @ theta.T.expand(self.n_dims,-1,-1)).sum(dim=-1)
    #     test = theta.unsqueeze(dim=1).expand(-1,self.n_dims,-1) * gamma
    #     nominator = test.unsqueeze(dim=-1).expand(-1,-1,-1,self.n_features) * seq
    #     denom = (gamma @ (theta.T @ theta)).sum(dim=-1, keepdim=True)
    #     # resulting shape of division shoudl be: (n_states, n_features, n_context) but wtf am i suppose to do with time dependence?
    #     return nominator / denom
    
    def _compute_covs(self, 
                     X:List[torch.Tensor],
                     gamma:List[torch.Tensor],
                     theta:Optional[ContextualVariables]) -> torch.Tensor:
        """Compute the covariances for each component."""
        new_covs = torch.zeros(size=(self.n_dims,self.n_features, self.n_features), 
                           dtype=torch.float64, 
                           device=self.device)
        
        denom = torch.zeros(size=(self.n_dims,1,1), 
                            dtype=torch.float64, 
                            device=self.device)

        for seq,gamma_val in zip(X,gamma):
            if theta is not None:
                # TODO: matmul shapes are inconsistent 
                raise NotImplementedError('Contextualized emissions not implemented for GaussianEmissions')
            else:
                gamma_expanded = gamma_val.T.unsqueeze(-1)
                diff = seq.expand(self.n_dims,-1,-1) - self.means.unsqueeze(1)
                new_covs += torch.transpose(diff * gamma_expanded,1,2) @ diff
                denom += torch.sum(gamma_expanded,dim=-2,keepdim=True)

        new_covs /= denom
        new_covs += self.min_covar * torch.eye(self.n_features, device=self.device)

        return new_covs
    
    def _update_emissions_params(self, 
                                X:List[torch.Tensor], 
                                gamma:List[torch.Tensor], 
                                theta:Optional[ContextualVariables]=None):
        self._means.copy_(self._compute_means(X,gamma,theta))
        self._covs.copy_(self._compute_covs(X,gamma,theta))