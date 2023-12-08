import torch
from abc import abstractmethod, abstractproperty, ABC
from typing import Optional, List
from ..utils import ContextualVariables


class BaseEmission(ABC):

    def __init__(self,
                 n_dims: int,
                 n_features: int,
                 discrete: bool,
                 seed:Optional[int] = None,
                 device:Optional[torch.device] = None):

        self.n_dims = n_dims
        self.n_features = n_features
        self.discrete = discrete
        self.seed = seed
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') if device is None else device
        
    @abstractproperty
    def __str__(self):
        pass

    @abstractproperty
    def pdf(self):
        pass

    @abstractmethod
    def map_emission(self, x:torch.Tensor) -> torch.Tensor:
        """Convert emissions into log probabilities."""
        pass

    @abstractmethod
    def sample_emissions_params(self, X:Optional[torch.Tensor]=None, seed:Optional[int]=None):
        """Sample emission parameters."""
        pass

    @abstractmethod
    def update_emission_params(self, X:List[torch.Tensor], gamma:List[torch.Tensor], theta:Optional[ContextualVariables]=None):
        """Update emission parameters in the model."""
        pass