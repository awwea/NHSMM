import torch
from torch.distributions import Categorical
from typing import Union, Tuple, Literal
from .utils import sample_logits, validate_logits # type: ignore


class StochasticTensor:
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
    """

    MAT_TYPES_HINT = Literal['Transition','Emission','Duration','Weights','Matrix','Tensor','Initial','Vector']
    
    def __init__(self,
                 logits:torch.Tensor,
                 name:MAT_TYPES_HINT):
        
        self.name = name
        self.logits = logits

    def __str__(self):
        return f"{self.__class__.__name__}(kind = {self.name}\nsize = {tuple(self.shape)})"

    @property
    def logits(self) -> torch.Tensor:
        return self._logits

    @logits.setter
    def logits(self, log_probs:torch.Tensor):
        self._logits = validate_logits(log_probs, self.name)

    @classmethod
    def from_dirichlet(cls, 
                       name:MAT_TYPES_HINT, 
                       size:Union[Tuple[int,...],torch.Size], 
                       semi_markov:bool = False,
                       prior:float = 1.0):
        matrix = sample_logits(prior,size,semi_markov)
        return cls(matrix,name)
        