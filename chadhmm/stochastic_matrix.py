import torch
import torch.nn as nn
from torch.distributions import Categorical
from typing import Union, Tuple, Literal
from .utils import sample_logits, validate_logits # type: ignore


class StochasticTensor(nn.Module):
    """
    Class for constructing the Probability Tensors using PyTorch Categorical distribution. Internally this only stores the logits 
    aka log probabilities that are exposed to the Categorical distribution.

    Parameters
    -----------
    logits (torch.Tensor):
        Logits represent the probability matrix where first dimension is also batch shape. The second dimension is number of categories for Categorical dist.
    name (MAT_TYPES_HINT):
        One of 'Transition','Emission','Duration','Weights','Matrix','Tensor','Initial' or 'Vector'
        This allows for validation of shapes and also identification.
    device (torch.device):
        On which device to move the Tensor.
    """

    MAT_TYPES_HINT = Literal['Transition','Emission','Duration','Weights','Matrix','Tensor','Initial','Vector']
    
    def __init__(self,
                 logits:torch.Tensor,
                 name:MAT_TYPES_HINT):
        
        super().__init__()
        self.name = name
        self.logits = logits

    def __str__(self):
        return f"{self.name}(shape={self.shape})"

    @property
    def logits(self) -> torch.Tensor:
        return self.param.data

    @logits.setter
    def logits(self, log_probs:torch.Tensor):
        self.param = nn.Parameter(validate_logits(log_probs,self.name),requires_grad=False)
        self.register_parameter(self.name, self.param)

    @property
    def pmf(self) -> Categorical:
        return Categorical(logits=self.param)
    
    @property
    def shape(self) -> torch.Size:
        return self.param.shape
    
    @property
    def probs(self) -> torch.Tensor:
        return self.logits.exp()

    @classmethod
    def from_dirichlet(cls, 
                       name:MAT_TYPES_HINT, 
                       size:Union[Tuple[int,...],torch.Size], 
                       semi_markov:bool = False,
                       prior:float = 1.0):
        matrix = sample_logits(prior,size,semi_markov)
        return cls(matrix,name)
        