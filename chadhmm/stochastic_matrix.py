import torch
import numpy as np
from torch.distributions import Categorical
from typing import Optional, Union, TypeVar, Tuple, Literal
from .utils import sample_logits, validate_logits # type: ignore

torch.set_printoptions(precision=4, profile="full")
MAT_OPS = TypeVar('MAT_OPS', bound=Union['StochasticTensor', np.ndarray, torch.Tensor])


class StochasticTensor:
    """
    Class for constructing the Probability Tensors using PyTorch Categorical distribution. Internally this only stores the logits 
    aka log probabilities that are exposed to the Categorical distribution.

    Parameters
    -----------
    logits (torch.Tensor):
        Logits represent the probability matrix where first dimension is also batch shape. The second dimension is number of categories for Categorical dist.
    name (MAT_TYPES_HINT):
        One of 'Transition','Emission','Duration','Weights','Matrix','Tensor','Initial' or 'Vector' - this allows for validation of shapes and also identification.
    device (torch.device):
        On which device to move the Tensor.
    """

    MAT_TYPES_HINT = Literal['Transition','Emission','Duration','Weights','Matrix','Tensor','Initial','Vector']
    
    def __init__(self,
                 logits:torch.Tensor,
                 name:MAT_TYPES_HINT,
                 device:Optional[torch.device] = None):
        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') if device is None else device
        self._logits = validate_logits(logits,name).to(self.device)
        self._name = name

    def __str__(self):
        return f"{self._name}(shape={self.shape})"

    @classmethod
    def from_dirichlet(cls, 
                       name:MAT_TYPES_HINT, 
                       size:Union[Tuple[int,...],torch.Size], 
                       device:torch.device,
                       semi_markov:bool = False,
                       prior:float = 1.0):
        matrix = sample_logits(prior,size,device,semi_markov)
        return cls(matrix,name,device)

    @property
    def pmf(self) -> Categorical:
        return Categorical(logits=self.logits)
    
    @property
    def shape(self) -> torch.Size:
        return self.pmf.param_shape
    
    @property
    def probs(self) -> torch.Tensor:
        return self._logits.exp()

    @property
    def logits(self) -> torch.Tensor:
        return self._logits

    @logits.setter
    def logits(self, log_probs:MAT_OPS):
        assert self.shape == log_probs.shape, ValueError("Matrix must be the same shape as the original")        
        if isinstance(log_probs, StochasticTensor):
            self._logits.copy_(log_probs.logits)
        elif isinstance(log_probs, torch.Tensor):
            self._logits.copy_(log_probs)
        elif isinstance(log_probs, np.ndarray):
            self._logits.copy_(torch.from_numpy(log_probs))
        else:
            raise NotImplementedError(f'Matrix type not supported, got {type(log_probs)}')
        
    def __mul__(self, other: MAT_OPS) -> torch.Tensor:
        if isinstance(other, StochasticTensor):
            return self.logits * other.logits
        elif isinstance(other, torch.Tensor):
            return self.logits * other
        elif isinstance(other, np.ndarray):
            return self.logits * torch.from_numpy(other)
        else:
            raise NotImplementedError(f'Unexpected type for operation, got {type(other)}')
        
    def __rmul__(self, other: MAT_OPS) -> torch.Tensor:
        return self.__mul__(other)
    
    def __matmul__(self, other: MAT_OPS) -> torch.Tensor:
        if isinstance(other, StochasticTensor):
            return self.logits @ other.logits
        elif isinstance(other, torch.Tensor):
            return self.logits @ other
        elif isinstance(other, np.ndarray):
            return self.logits @ torch.from_numpy(other)
        else:
            raise NotImplementedError(f'Unexpected type for operation, got {type(other)}')
    
    def __rmatmul__(self, other: MAT_OPS) -> torch.Tensor:
        return self.__matmul__(other)

    def __truediv__(self, other: MAT_OPS) -> torch.Tensor:
        if isinstance(other, StochasticTensor):
            return self.logits / other.logits
        elif isinstance(other, torch.Tensor):
            return self.logits / other
        elif isinstance(other, np.ndarray):
            return self.logits / torch.from_numpy(other)
        else:
            raise NotImplementedError(f'Unexpected type for operation, got {type(other)}')
    
    def __rtruediv__(self, other: MAT_OPS) -> torch.Tensor:
        return self.__truediv__(other)

    def __add__(self, other: MAT_OPS) -> torch.Tensor:
        if isinstance(other, StochasticTensor):
            return self.logits + other.logits
        elif isinstance(other, torch.Tensor):
            return self.logits + other
        elif isinstance(other, np.ndarray):
            return self.logits + torch.from_numpy(other)
        else:
            raise NotImplementedError(f'Unexpected type for operation, got {type(other)}')
        
    def __radd__(self, other: MAT_OPS) -> torch.Tensor:
        return self.__add__(other)
    
    def __sub__(self, other: MAT_OPS) -> torch.Tensor:
        if isinstance(other, StochasticTensor):
            return self.logits - other.logits
        elif isinstance(other, torch.Tensor):
            return self.logits - other
        elif isinstance(other, np.ndarray):
            return self.logits - torch.from_numpy(other)
        else:
            raise NotImplementedError(f'Unexpected type for operation, got {type(other)}')
        
    def __rsub__(self, other: MAT_OPS) -> torch.Tensor:
        return self.__sub__(other)

    def __eq__(self, other: MAT_OPS) -> bool: #type: ignore
        if isinstance(other, StochasticTensor):
            return torch.equal(self.logits, other.logits)
        elif isinstance(other, torch.Tensor):
            return torch.equal(self.logits, other)
        elif isinstance(other, np.ndarray):
            return torch.equal(self.logits, torch.from_numpy(other))
        else:
            raise NotImplementedError(f'Unexpected type for operation, got {type(other)}')
    
    def __len__(self) -> int:
        return self.logits.numel()
    
    def __getitem__(self, idx) -> torch.Tensor:
        if isinstance(idx, tuple):
            row_idx, col_idx = idx
            if isinstance(row_idx, slice) or isinstance(col_idx, slice):
                return self.logits[row_idx, col_idx]
            else:
                return self.logits[row_idx, col_idx]
        else:
            return self.logits[idx]
    
    def __setitem__(self, row_idx, col_idx, value):
        self.logits[row_idx, col_idx] = value        

        