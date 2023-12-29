from abc import ABC, abstractmethod, abstractproperty
from typing import Optional, List, Tuple, Dict, Sequence , Literal, Union

import torch
import numpy as np

from ..stochastic_matrix import StochasticTensor # type: ignore
from ..utils import (Multiprocessor, FittedModel, ContextualVariables, ConvergenceHandler, Observations, SeedGenerator,
log_normalize, sequence_generator, 
DECODERS, INFORM_CRITERIA) # type: ignore


class BaseHSMM(ABC):
    """
    Base Class for Hidden Semi-Markov Model (HSMM)
    ----------
    A Hidden Semi-Markov Model (HSMM) subclass that provides a foundation for building specific HMM models. HSMM is not assuming that the duration of each state is geometrically distributed, 
    but rather that it is distributed according to a general distribution. This duration is also reffered to as the sojourn time.
    """
    def __init__(self,
                 n_states: int,
                 max_duration: int,
                 alpha: float = 1.0,
                 seed: Optional[int] = None,
                 device: Optional[torch.device] = None):

        self.n_states = n_states
        self.max_duration = max_duration
        self.alpha = alpha
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') if device is None else device

        self._seed_gen = SeedGenerator(seed)
        self._initial_vector, self._transition_matrix, self._duration_matrix = self.sample_chain_params(self.alpha)

    @property
    def seed(self):
        self._seed_gen.seed

    @property
    def duration_matrix(self) -> StochasticTensor:
        return self._duration_matrix

    @duration_matrix.setter
    def duration_matrix(self, matrix):
        self._duration_matrix.logits = matrix
        
    @property
    def transition_matrix(self) -> StochasticTensor:
        return self._transition_matrix

    @transition_matrix.setter
    def transition_matrix(self, matrix):
        self._transition_matrix.logits = matrix

    @property
    def initial_vector(self) -> StochasticTensor:
        return self._initial_vector

    @initial_vector.setter
    def initial_vector(self, vector):
        self._initial_vector.logits = vector
        
    @property
    def _check_params(self):
        """Check if the model parameters are set."""
        return self.params

    @abstractproperty
    def params(self) -> Dict[str, torch.Tensor]:
        """Returns the parameters of the model."""
        pass

    @abstractproperty   
    def n_fit_params(self) -> Dict[str, int]:
        """Return the number of trainable model parameters."""
        pass
    
    @abstractproperty
    def dof(self) -> int:
        """Returns the degrees of freedom of the model."""
        pass

    @abstractmethod
    def map_emission(self, x:torch.Tensor) -> torch.Tensor:
        """Get emission probabilities for a given sequence of observations."""
        pass

    @abstractmethod
    def check_sequence(self, X:torch.Tensor) -> torch.Tensor:
        """Check if the sequence is valid, encode, transform if necessary."""
        pass

    @abstractmethod
    def sample_B_params(self, X:Optional[torch.Tensor], seed:Optional[int] = None):
        """Sample the emission parameters."""
        pass

    @abstractmethod
    def update_B_params(self, X:List[torch.Tensor], log_gamma:List[torch.Tensor], theta:torch.Tensor):
        """Update the emission parameters."""
        pass

    def sample_chain_params(self, alpha:float = 1) -> Tuple[StochasticTensor, StochasticTensor, StochasticTensor]:
        """Initialize the parameters of Semi-Markov Chain."""
        return (StochasticTensor.from_dirichlet('Initial',(self.n_states,),self.device,False,alpha), 
                StochasticTensor.from_dirichlet('Transition',(self.n_states,self.n_states),self.device,True,alpha),
                StochasticTensor.from_dirichlet('Duration',(self.n_states,self.max_duration),self.device,False,alpha))
    
    def to_observations(self, X:torch.Tensor, lengths:Optional[List[int]]=None) -> Observations:
        """Convert a sequence of observations to an Observations object."""
        X_valid = self.check_sequence(X)
        n_samples = X_valid.shape[0]
        seq_lenghts = [n_samples] if lengths is None else lengths
        X_vec = list(torch.split(X_valid, seq_lenghts))

        log_probs = []
        for seq in X_vec:
            log_probs.append(self.map_emission(seq))
        
        return Observations(n_samples,X_vec,log_probs,seq_lenghts,len(seq_lenghts))  

    def to_contextuals(self, theta:torch.Tensor, X:Observations) -> ContextualVariables:
        """Returns the parameters of the model."""
        if (n_dim:=theta.ndim) != 2:
            raise ValueError(f'Context must be 2-dimensional. Got {n_dim}.')
        elif theta.shape[1] not in (1, X.n_samples):
            raise ValueError(f'Context must have shape (context_vars, 1) for time independent context or (context_vars,{X.n_samples}) for time dependent. Got {theta.shape}.')
        else:
            n_context, n_observations = theta.shape
            time_dependent = n_observations == X.n_samples
            adj_theta = torch.vstack((theta, torch.ones(size=(1,n_observations),
                                                        dtype=torch.float64,
                                                        device=self.device)))
            if not time_dependent:
                adj_theta = adj_theta.expand(n_context+1, X.n_samples)

            context_matrix = list(torch.split(adj_theta,X.lengths,1))
            return ContextualVariables(n_context, context_matrix, time_dependent) 

    def fit(self,
            X:torch.Tensor,
            tol:float = 1e-2,
            max_iter:int = 20,
            n_init:int = 1,
            post_conv_iter:int = 3,
            ignore_conv:bool = False,
            sample_B_from_X:bool = False,
            verbose:bool = True,
            plot_conv:bool = False,
            lengths:Optional[List[int]] = None,
            theta:Optional[torch.Tensor] = None) -> Dict[int, FittedModel]:
        """Fit the model to the given sequence using the EM algorithm."""
        if sample_B_from_X:
            self.sample_B_params(X)
        X_valid = self.to_observations(X,lengths)
        valid_theta = self.to_contextuals(theta,X_valid) if theta is not None else None

        self.conv = ConvergenceHandler(tol=tol,
                                       max_iter=max_iter,
                                       n_init=n_init,
                                       post_conv_iter=post_conv_iter,
                                       device=self.device,
                                       verbose=verbose)

        self._check_params
        distinct_models = {}
        for rank in range(n_init):
            if rank > 0:
                self._initial_vector, self._transition_matrix, self._duration_matrix = self.sample_chain_params()
            
            self.conv.push_pull(sum(self._compute_log_likelihood(X_valid)),0,rank)
            for iter in range(1,self.conv.max_iter+1):
                # EM algorithm step
                self._update_model(X_valid, valid_theta)

                # remap emission probabilities after update of B
                X_valid.log_probs = [self.map_emission(x) for x in X_valid.X]
                
                curr_log_like = sum(self._compute_log_likelihood(X_valid))
                converged = self.conv.push_pull(curr_log_like,iter,rank)
                if converged and not ignore_conv:
                    print(f'Model converged after {iter} iterations with log-likelihood: {curr_log_like:.2f}')
                    break

            distinct_models[rank] = FittedModel(self.__str__(),
                                                self.n_fit_params, 
                                                self.dof,
                                                converged,
                                                curr_log_like, 
                                                self.ic(X,lengths),
                                                self.params)
        
        if plot_conv:
            self.conv.plot_convergence()

        return distinct_models
    
    def predict(self, 
                X:torch.Tensor, 
                lengths:Optional[List[int]] = None,
                algorithm:Literal['map','viterbi'] = 'viterbi') -> Tuple[float, Sequence[torch.Tensor]]:
        """Predict the most likely sequence of hidden states."""

        if algorithm not in DECODERS:
            raise ValueError(f'Unknown decoder algorithm {algorithm}')
        
        decoder = {'viterbi': self._viterbi,
                   'map': self._map}[algorithm]
        
        self._check_params
        X_valid = self.to_observations(X, lengths)
        log_score = sum(self._compute_log_likelihood(X_valid))
        decoded_path = decoder(X_valid)

        return log_score, decoded_path

    def score(self, X:torch.Tensor, lengths:Optional[List[int]]=None) -> float:
        """Compute the joint log-likelihood"""
        return sum(self._score_observations(X,lengths))
    
    def score_samples(self, X:torch.Tensor, lengths:Optional[List[int]]=None) -> List[float]:
        """Compute the log-likelihood for each sequence"""
        return self._score_observations(X,lengths)

    def ic(self, 
           X:torch.Tensor, 
           lengths:Optional[List[int]] = None, 
           criterion:Literal['AIC','BIC','HQC'] = 'AIC') -> float:
        """Calculates the information criteria for a given model."""
        log_likelihood = self.score(X, lengths)

        if criterion not in INFORM_CRITERIA:
            raise NotImplementedError(f'{criterion} is not a valid information criterion. Valid criteria are: {INFORM_CRITERIA}')
        
        criterion_compute = {'AIC': lambda log_likelihood, dof: -2.0 * log_likelihood + 2.0 * dof,
                             'BIC': lambda log_likelihood, dof: -2.0 * log_likelihood + dof * np.log(X.shape[0]),
                             'HQC': lambda log_likelihood, dof: -2.0 * log_likelihood + 2.0 * dof * np.log(np.log(X.shape[0]))}[criterion]
        
        return criterion_compute(log_likelihood, self.dof)

    def _forward(self, X:Observations) -> List[torch.Tensor]:
        """Forward pass of the forward-backward algorithm."""
        alpha_vec = []
        for seq_len,_,log_probs in sequence_generator(X):
            log_alpha = torch.zeros(size=(seq_len,self.n_states,self.max_duration),
                                    dtype=torch.float64,
                                    device=self.device)
            
            log_alpha[0] = self._duration_matrix + (self._initial_vector + log_probs[0]).reshape(-1,1)
            for t in range(1,seq_len):
                trans_alpha_sum = torch.logsumexp(log_alpha[t-1,:,0].reshape(-1,1) + self._transition_matrix, dim=0) + log_probs[t]

                log_alpha[t,:,-1] = trans_alpha_sum + self._duration_matrix[:,-1]
                log_alpha[t,:,:-1] = torch.logaddexp(log_alpha[t-1,:,1:] + log_probs[t].reshape(-1,1),
                                                     trans_alpha_sum.reshape(-1,1) + self._duration_matrix[:,:-1])
            
            alpha_vec.append(log_alpha)
        
        return alpha_vec

    def _backward(self, X:Observations) -> List[torch.Tensor]:
        """Backward pass of the forward-backward algorithm."""
        beta_vec = []
        for seq_len,_,log_probs in sequence_generator(X):
            log_beta = torch.zeros(size=(seq_len,self.n_states,self.max_duration),
                                   dtype=torch.float64,
                                   device=self.device)
            
            for t in reversed(range(seq_len-1)):
                beta_dur_sum = torch.logsumexp(log_beta[t+1] + self._duration_matrix, dim=1)

                log_beta[t,:,0] = torch.logsumexp(self._transition_matrix + log_probs[t+1] + beta_dur_sum, dim=1)
                log_beta[t,:,1:] = log_probs[t+1].reshape(-1,1) + log_beta[t+1,:,:-1]
            
            beta_vec.append(log_beta)

        return beta_vec

    def _gamma(self, X:Observations, log_alpha:List[torch.Tensor], log_xi:List[torch.Tensor]) -> List[torch.Tensor]:
        """Compute the log-Gamma variable in Hidden Markov Model."""
        gamma_vec = []
        for (seq_len,_,_),alpha,xi in zip(sequence_generator(X),log_alpha,log_xi):
            log_gamma = torch.zeros(size=(seq_len,self.n_states), 
                                    dtype=torch.float64, 
                                    device=self.device)

            real_xi = xi.exp()
            log_gamma[-1] = alpha[-1].logsumexp(dim=1)
            for t in reversed(range(seq_len-1)):
                log_gamma[t] = torch.log(log_gamma[t+1].exp() + torch.sum(real_xi[t] - real_xi[t].transpose(-2,-1),dim=1))

            log_gamma -= alpha[-1].logsumexp(dim=(0,1))
            gamma_vec.append(log_gamma)

        return gamma_vec

    def _xi(self, X:Observations, log_alpha:List[torch.Tensor], log_beta:List[torch.Tensor]) -> List[torch.Tensor]:
        """Compute the log-Xi variable in Hidden Markov Model."""
        xi_vec = []
        for (seq_len,_,log_probs),alpha,beta in zip(sequence_generator(X),log_alpha,log_beta):
            log_xi = torch.zeros(size=(seq_len-1,self.n_states,self.n_states),
                                   dtype=torch.float64, 
                                   device=self.device)
            
            for t in range(seq_len-1):
                beta_dur_sum = torch.logsumexp(beta[t+1] + self._duration_matrix, dim=1)
                log_xi[t] = alpha[t,:,0].reshape(-1,1) + self._transition_matrix + log_probs[t+1] + beta_dur_sum

            xi_vec.append(log_xi)

        return xi_vec
    
    def _eta(self, X:Observations, log_alpha:List[torch.Tensor], log_beta:List[torch.Tensor]) -> List[torch.Tensor]:
        """Compute the Eta variable in Hidden Markov Model."""
        eta_vec = []
        for (seq_len,_,log_probs),alpha,beta in zip(sequence_generator(X),log_alpha,log_beta):
            log_eta = torch.zeros(size=(seq_len-1,self.n_states,self.max_duration), 
                                  dtype=torch.float64, 
                                  device=self.device)
            
            for t in range(seq_len-1):
                trains_alpha_sum = torch.logsumexp(alpha[t,:,0].reshape(-1,1) + self._transition_matrix, dim=0)
                log_eta[t] = beta[t+1] + self._duration_matrix + (log_probs[t+1] + trains_alpha_sum).reshape(-1, 1) 

            log_eta -= alpha[-1].logsumexp(dim=(0,1))
            eta_vec.append(log_eta)
        
        return eta_vec
    
    def _compute_fwd_bwd(self, X:Observations) -> Tuple[List[torch.Tensor],List[torch.Tensor]]:
        log_alpha = self._forward(X)
        log_beta = self._backward(X)
        return log_alpha, log_beta

    def _compute_posteriors(self, X:Observations) -> Tuple[List[torch.Tensor],List[torch.Tensor],List[torch.Tensor]]:
        """Execute the forward-backward algorithm and compute the log-Gamma, log-Xi and Log-Eta variables."""
        log_alpha, log_beta = self._compute_fwd_bwd(X)
        log_xi = self._xi(X, log_alpha, log_beta)
        log_eta = self._eta(X, log_alpha, log_beta)
        log_gamma = self._gamma(X, log_alpha, log_xi)

        # TODO: Normalize the Xi after being used in gamma estimation
        for i,(xi,alpha) in enumerate(zip(log_xi,log_alpha)):
            log_xi[i] = xi - alpha[-1].logsumexp(dim=(0,1))

        return log_gamma, log_xi, log_eta
    
    def _accum_pi(self, log_gamma:List[torch.Tensor]) -> torch.Tensor:
        """Accumulate the statistics for the initial vector."""
        log_pi = torch.zeros(size=(self.n_states,),
                             dtype=torch.float64, 
                             device=self.device)

        for gamma in log_gamma:
            log_pi += gamma[0].exp()
        
        return log_normalize(log_pi.log(),0)

    def _accum_A(self, log_xi:List[torch.Tensor]) -> torch.Tensor:
        """Accumulate the statistics for the transition matrix."""
        log_A = torch.zeros(size=(self.n_states,self.n_states),
                             dtype=torch.float64, 
                             device=self.device)
        
        for xi in log_xi:
            log_A += xi.exp().sum(dim=0)

        return log_normalize(log_A.log(),1)

    def _accum_D(self, log_eta:List[torch.Tensor]) -> torch.Tensor:
        """Accumulate the statistics for the transition matrix."""
        log_D = torch.zeros(size=(self.n_states,self.max_duration),
                             dtype=torch.float64, 
                             device=self.device)
        
        for eta in log_eta:
            log_D += eta.exp().sum(dim=0)

        return log_normalize(log_D.log(),1)

    def _update_model(self, X:Observations, theta):
        """Compute the updated parameters for the model."""
        log_gamma, log_xi, log_eta = self._compute_posteriors(X)

        self._initial_vector._logits.copy_(self._accum_pi(log_gamma))
        self._transition_matrix._logits.copy_(self._accum_A(log_xi))
        self._duration_matrix._logits.copy_(self._accum_D(log_eta))
        self.update_B_params(X.X,log_gamma,theta)

    def _viterbi(self, X:Observations) -> Sequence[torch.Tensor]:
        """Viterbi algorithm for decoding the most likely sequence of hidden states."""
        raise NotImplementedError('Viterbi algorithm not yet implemented for HSMM')

    def _map(self, X:Observations) -> Sequence[torch.Tensor]:
        """Compute the most likely (MAP) sequence of indiviual hidden states."""
        map_paths = []
        gamma_vec,_,_ = self._compute_posteriors(X)
        for gamma in gamma_vec:
            map_paths.append(torch.argmax(gamma,dim=1))
        
        return map_paths

    def _score_observations(self, X:torch.Tensor, lengths:Optional[List[int]]=None) -> List[float]:
        """Compute the log-likelihood for each sample sequence."""
        self._check_params
        return self._compute_log_likelihood(self.to_observations(X, lengths))

    def _compute_log_likelihood(self, X:Observations) -> List[float]:
        """Compute the log-likelihood of the given sequence."""
        scores = []
        for alpha in self._forward(X):
            scores.append(alpha[-1].logsumexp(dim=(0,1)).item())

        return scores
