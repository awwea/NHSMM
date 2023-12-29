import torch
import numpy as np
from typing import Optional, List, Literal, Dict
from .emissions import GaussianMixtureEmissions
from .utils import FittedModel, ConvergenceHandler, INFORM_CRITERIA


class GaussianMixtureModel(GaussianMixtureEmissions):

    def __init__(self, 
                 n_dims:int,
                 n_components:int,
                 n_features:int,
                 alpha:float = 1.0,
                 covariance_type:GaussianMixtureEmissions.COVAR_TYPES_HINT = 'full',
                 min_covar:float = 1e-3,
                 seed:Optional[int] = None,
                 device:Optional[torch.device] = None):
        
        GaussianMixtureEmissions.__init__(self,n_dims,n_features,n_components,False,alpha,covariance_type,min_covar,seed,device)
    
    @property
    def dof(self) -> int:
        return self.n_dims*(self.n_components-1) + self.n_dims*self.n_features*self.n_components + self.n_dims*self.n_features*(self.n_features + 1) // 2
    
    @property
    def n_fit_params(self) -> Dict[str,int]:
        return {'weights': self.n_dims * self.n_components,
                'means': self.n_dims * self.n_features * self.n_components,
                'covs': {
                    'spherical': self.n_dims,
                    'diag': self.n_dims * self.n_features,
                    'full': self.n_dims * self.n_features * (self.n_features + 1) // 2,
                    'tied': self.n_features * (self.n_features + 1) // 2,
                }[self.covariance_type]}
    
    @property
    def params(self):
        return {'weights':self.weights.logits,
                'means':self.means,
                'covars':self.covs}

    def score(self,X:torch.Tensor) -> float:
        return self.map_emission(X).sum(0).item()

    def ic(self, 
           X:torch.Tensor, 
           criterion:Literal['AIC','BIC','HQC'] = 'AIC') -> float:
        """Calculates the information criteria for a given model."""
        log_likelihood = self.score(X)
        if criterion not in INFORM_CRITERIA:
            raise NotImplementedError(f'{criterion} is not a valid information criterion. Valid criteria are: {INFORM_CRITERIA}')
        
        criterion_compute = {'AIC': lambda log_likelihood, dof: -2.0 * log_likelihood + 2.0 * dof,
                             'BIC': lambda log_likelihood, dof: -2.0 * log_likelihood + dof * np.log(X.shape[0]),
                             'HQC': lambda log_likelihood, dof: -2.0 * log_likelihood + 2.0 * dof * np.log(np.log(X.shape[0]))}[criterion]
        
        return criterion_compute(log_likelihood, self.dof)

    def fit(self,
            X:torch.Tensor,
            tol:float=1e-2,
            max_iter:int=20,
            n_init:int=1,
            post_conv_iter:int=3,
            ignore_conv:bool=False,
            plot_conv:bool=False,
            verbose:bool=False,
            lengths:Optional[List[int]] = None,
            theta:Optional[torch.Tensor] = None) -> Dict[int,FittedModel]:
        """Fit the model to the data."""
        X_valid = GaussianMixtureEmissions.check_constraints(self,X)
        X_vec = list(torch.split(X_valid, lengths)) if lengths is not None else [X_valid]

        self.conv = ConvergenceHandler(tol=tol,
                                       max_iter=max_iter,
                                       n_init=n_init,
                                       post_conv_iter=post_conv_iter,
                                       device=self.device,
                                       verbose=verbose)

        distinct_models = {}
        for rank in range(n_init):
            self.conv.push_pull(self.score(X),0,rank)
            for iter in range(1,self.conv.max_iter+1):
                # EM algorithm step
                self.update_emission_params(X_vec,self._compute_responsibilities(X_vec),theta)
                
                curr_log_like = self.score(X)
                converged = self.conv.push_pull(curr_log_like,iter,rank)
                if converged and not ignore_conv:
                    print(f'Model converged after {iter} iterations with log-likelihood: {curr_log_like:.2f}')
                    break

            distinct_models[rank] = FittedModel(self.__str__(),
                                                self.n_fit_params, 
                                                self.dof,
                                                converged,
                                                curr_log_like, 
                                                self.ic(X),
                                                self.params)
        
        if plot_conv:
            self.conv.plot_convergence()

        return distinct_models