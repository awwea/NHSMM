import torch
from abc import ABC, abstractmethod, abstractproperty
from typing import Optional, List, Dict
from torch.distributions import Distribution, MixtureSameFamily
from ..stochastic_matrix import WeightsMatrix
from ..utils import ContextualVariables, log_normalize


class MixtureEmissions(ABC):
    """
    Mixture model for HMM emissions. This class is an abstract base class for Gaussian, Poisson and other mixture models.
    """

    def __init__(self, 
                 n_dims: int,
                 n_components: int,
                 n_features: int,
                 alpha: float = 1.0,
                 init_weights: bool = True,
                 seed: Optional[int] = None,
                 device: Optional[torch.device] = None):
        
        self.n_dims = n_dims
        self.alpha = alpha
        self.n_components = n_components
        self.n_features = n_features
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') if device is None else device

        if init_weights:
            self._weights = self.sample_weights(seed)

    @property
    def weights(self) -> WeightsMatrix:
        try:
            return self._weights
        except AttributeError:
            raise AttributeError('Weights are not initialized')

    @weights.setter
    def weights(self, vector):
        assert (self.n_dims, self.n_components) == vector.shape, 'Matrix dimensions differ from HMM model'
        if isinstance(vector, WeightsMatrix):
            self._weights = vector
        elif isinstance(vector, torch.Tensor):
            self._weights = WeightsMatrix(n_states=self.n_dims, 
                                          n_components=self.n_components,
                                          matrix=vector,
                                          device=self.device)
        else:
            raise NotImplementedError(f'Expected torch Tensor or WeightsMatrix object, got {type(vector)}')
        
    @property
    def mixture_pdf(self) -> MixtureSameFamily:
        """Return the emission distribution for Gaussian Mixture Distribution."""
        return MixtureSameFamily(mixture_distribution = self.weights._dist,
                                 component_distribution = self.pdf)

    @abstractproperty
    def pdf(self) -> Distribution:
        """Return the emission distribution of Mixture."""
        pass

    @abstractproperty
    def params(self) -> Dict[str, torch.Tensor]:
        """Return the parameters of the Mixture."""
        pass

    @abstractmethod
    def _update_params(self, 
                       X:List[torch.Tensor], 
                       resp:List[torch.Tensor], 
                       theta:Optional[ContextualVariables]=None) -> None:
        """Update the parameters of the Mixture."""
        pass 
    
    def sample_weights(self, 
                       seed:Optional[int]) -> WeightsMatrix:
        """Sample the weights for the mixture."""
        return WeightsMatrix(n_states=self.n_dims,
                             n_components=self.n_components,
                             rand_seed=seed,
                             alpha=self.alpha,
                             device=self.device)    

    def map_emission(self, 
                     x:torch.Tensor) -> torch.Tensor:
        x_batched = x.unsqueeze(1).expand(-1,self.n_dims,-1)
        return self.mixture_pdf.log_prob(x_batched)
    
    def _compute_responsibilities(self, X:List[torch.Tensor]) -> List[torch.Tensor]:
        """Compute the responsibilities for each component."""
        resp_vec = []
        for seq in X:
            n_observations = seq.size(dim=0)
            log_responsibilities = torch.zeros(size=(self.n_dims,self.n_components,n_observations), 
                                               dtype=torch.float64, 
                                               device=self.device)

            for t in range(n_observations):
                log_responsibilities[:,:,t] = log_normalize(self.weights.matrix + self.pdf.log_prob(seq[t]),1)

            resp_vec.append(log_responsibilities)
        
        return resp_vec
    
    def _compute_weights(self, posterior:List[torch.Tensor]) -> torch.Tensor:
        log_weights = torch.zeros(size=(self.n_dims,self.n_components),
                                  dtype=torch.float64, 
                                  device=self.device)

        for p in posterior:
            log_weights += p.exp().sum(-1)
        
        return log_normalize(log_weights.log(),1)