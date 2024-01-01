from typing import Optional, Sequence, List, Tuple, Literal
from abc import ABC, abstractmethod, abstractproperty

import torch
import torch.nn as nn
import numpy as np

from ..utils import (ContextualVariables, ConvergenceHandler, Observations, SeedGenerator,
log_normalize, sequence_generator, sample_logits,
DECODERS, INFORM_CRITERIA) # type: ignore


class BaseHMM(nn.Module,ABC):
    """
    Base Abstract Class for HMM
    ----------
    Base Class of Hidden Markov Models (HMM) class that provides a foundation for building specific HMM models.
    """

    def __init__(self,
                 n_states:int,
                 n_features:int,
                 alpha:float = 1.0,
                 seed:Optional[int] = None):

        nn.Module.__init__(self)
        self.n_states = n_states
        self.n_features = n_features
        self.alpha = alpha
        self._seed_gen = SeedGenerator(seed)
        self.params = self.sample_model_params(self.alpha)
        
    @property
    def seed(self):
        self._seed_gen.seed

    @abstractproperty
    def pdf(self):
        pass
    
    @abstractproperty
    def dof(self):
        """Returns the degrees of freedom of the model."""
        pass

    @abstractmethod
    def map_emission(self, x:torch.Tensor) -> torch.Tensor:
        """Get emission probabilities for a given sequence of observations."""
        pass

    @abstractmethod
    def estimate_emission_params(self, 
                                 X:List[torch.Tensor], 
                                 posterior:List[torch.Tensor], 
                                 theta:Optional[ContextualVariables]) -> nn.ParameterDict:
        """Update the emission parameters."""
        pass

    @abstractmethod
    def sample_emission_params(self, X:Optional[torch.Tensor]=None) -> nn.ParameterDict:
        """Sample the emission parameters."""
        pass

    def sample(self, size:Sequence[int]) -> torch.Tensor:
        """Sample from Markov chain, either 1D for a single sequence or 2D for multiple sample sequences given by 0 axis."""
        n_dim = len(size)
        if n_dim == 1:
            start_sample = torch.Size([1])
        elif n_dim == 2:
            start_sample = torch.Size([size[0],1])
        else:
            raise ValueError(f'Size must be at most 2-dimensional, got {n_dim}.')
        
        sampled_paths = torch.hstack((self.initial_vector.pmf.sample(start_sample), 
                                  torch.zeros(size,dtype=torch.int)))
        
        for row,step in enumerate(self.transition_matrix.pmf.sample(torch.Size(size))):
            sampled_paths[row+1] = step[sampled_paths[row]]

        return sampled_paths
        
    def sample_model_params(self, alpha:float = 1.0, X:Optional[torch.Tensor]=None) -> nn.ParameterDict:
        """Initialize the model parameters."""
        initial_vector = sample_logits(alpha,(self.n_states,),False)
        transition_matrix = sample_logits(alpha,(self.n_states,self.n_states),False)
        model_params = nn.ParameterDict({
            'pi':nn.Parameter(initial_vector,requires_grad=False),
            'A':nn.Parameter(transition_matrix,requires_grad=False)
        })
        model_params.update(self.sample_emission_params(X))

        return model_params
    
    # TODO: torch.distributions.Distribution method _validate_sample does this but just on single batch
    def check_constraints(self,
                          value:torch.Tensor) -> torch.Tensor:
        not_supported = value[torch.logical_not(self.pdf.support.check(value))].unique()
        events = self.pdf.event_shape
        event_dims = len(events)
        assert len(not_supported) == 0, ValueError(f'Values outside PDF support, got values: {not_supported.tolist()}')
        assert value.ndim == event_dims+1, ValueError(f'Expected number of dims differs from PDF constraints on event shape {events}')
        if event_dims > 0:
            assert value.shape[1:] == events, ValueError(f'PDF event shape differs, expected {events} but got {value.shape[1:]}')
        return value

    def to_observations(self, X:torch.Tensor, lengths:Optional[List[int]]=None) -> Observations:
        """Convert a sequence of observations to an Observations object."""
        X_valid = self.check_constraints(X)
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
                                                        dtype=torch.float64)))
            if not time_dependent:
                adj_theta = adj_theta.expand(n_context+1, X.n_samples)

            context_matrix = list(torch.split(adj_theta,X.lengths,1))
            return ContextualVariables(n_context, context_matrix, time_dependent) 

    def fit(self,
            X:torch.Tensor,
            tol:float=1e-2,
            max_iter:int=20,
            n_init:int=1,
            post_conv_iter:int=3,
            ignore_conv:bool=False,
            sample_B_from_X:bool=False,
            verbose:bool=True,
            plot_conv:bool=False,
            lengths:Optional[List[int]]=None,
            theta:Optional[torch.Tensor]=None):
        """Fit the model to the given sequence using the EM algorithm."""
        if sample_B_from_X:
            self.sample_emission_params(X)
        X_valid = self.to_observations(X,lengths)
        valid_theta = self.to_contextuals(theta,X_valid) if theta is not None else None

        self.conv = ConvergenceHandler(tol=tol,
                                       max_iter=max_iter,
                                       n_init=n_init,
                                       post_conv_iter=post_conv_iter,
                                       verbose=verbose)

        for rank in range(n_init):
            if rank > 0:
                self.params.update(self.sample_model_params(self.alpha,X))
            
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
        
        if plot_conv:
            self.conv.plot_convergence()

        return self

    def predict(self, 
                X:torch.Tensor, 
                lengths:Optional[List[int]] = None,
                algorithm:Literal['map','viterbi'] = 'viterbi') -> Tuple[List[float],Sequence[torch.Tensor]]:
        """Predict the most likely sequence of hidden states. Returns log-likelihood and sequences"""
        if algorithm not in DECODERS:
            raise ValueError(f'Unknown decoder algorithm {algorithm}')
        
        decoder = {'viterbi': self._viterbi,
                   'map': self._map}[algorithm]
        
        X_valid = self.to_observations(X, lengths)
        log_score = self._compute_log_likelihood(X_valid)
        decoded_path = decoder(X_valid)

        return log_score, decoded_path

    def score(self, 
              X:torch.Tensor,
              by_sample:bool=True,
              lengths:Optional[List[int]]=None) -> List[float]:
        """Compute the joint log-likelihood"""
        log_likelihoods = self._compute_log_likelihood(self.to_observations(X, lengths))
        res = log_likelihoods if by_sample else [sum(log_likelihoods)]
        return res

    def ic(self,
           X:torch.Tensor,
           by_sample:bool=True,
           lengths:Optional[List[int]] = None,
           criterion:Literal['AIC','BIC','HQC'] = 'AIC') -> List[float]:
        """Calculates the information criteria for a given model."""
        log_likelihood = self.score(X,by_sample,lengths)
        if criterion not in INFORM_CRITERIA:
            raise NotImplementedError(f'{criterion} is not a valid information criterion. Valid criteria are: {INFORM_CRITERIA}')
        
        criterion_compute = {'AIC': lambda log_likelihood, dof: -2.0 * log_likelihood + 2.0 * dof,
                             'BIC': lambda log_likelihood, dof: -2.0 * log_likelihood + dof * np.log(X.shape[0]),
                             'HQC': lambda log_likelihood, dof: -2.0 * log_likelihood + 2.0 * dof * np.log(np.log(X.shape[0]))}[criterion]
        
        log_like_samples = []
        for sample_log_like in log_likelihood:
            log_like_samples.append(criterion_compute(sample_log_like, self.dof))

        return log_like_samples
    
    def _forward(self, X:Observations) -> List[torch.Tensor]:
        """Forward pass of the forward-backward algorithm."""
        alpha_vec = []
        for seq_len,_,log_probs in sequence_generator(X):
            log_alpha = torch.zeros(size=(seq_len,self.n_states), 
                                    dtype=torch.float64)
            
            log_alpha[0] = self.params.pi + log_probs[0]
            for t in range(1,seq_len):
                log_alpha[t] = torch.logsumexp(log_alpha[t-1].reshape(-1,1) + self.params.A, dim=0) + log_probs[t]

            alpha_vec.append(log_alpha)

        return alpha_vec
    
    def _backward(self, X:Observations) -> List[torch.Tensor]:
        """Backward pass of the forward-backward algorithm."""
        beta_vec = []
        for seq_len,_,log_probs in sequence_generator(X):
            log_beta = torch.zeros(size=(seq_len,self.n_states), 
                               dtype=torch.float64)
            
            for t in reversed(range(seq_len-1)):
                log_beta[t] = torch.logsumexp(self.params.A + log_probs[t+1] + log_beta[t+1], dim=1)
            
            beta_vec.append(log_beta)

        return beta_vec
    
    def _gamma(self, log_alpha:List[torch.Tensor], log_beta:List[torch.Tensor]) -> List[torch.Tensor]:
        """Compute the log-Gamma variable in Hidden Markov Model."""
        gamma_vec = []
        for alpha,beta in zip(log_alpha,log_beta):
            gamma_vec.append(log_normalize(alpha+beta,1))

        return gamma_vec
    
    def _xi(self, X:Observations, log_alpha:List[torch.Tensor], log_beta:List[torch.Tensor]) -> List[torch.Tensor]:
        """Compute the log-Xi variable in Hidden Markov Model."""
        xi_vec = []
        for (seq_len,_,log_probs),alpha,beta in zip(sequence_generator(X),log_alpha,log_beta):
            log_xi = torch.zeros(size=(seq_len-1, self.n_states, self.n_states),
                                 dtype=torch.float64)
            
            for t in range(seq_len-1):
                log_xi[t] = alpha[t].reshape(-1,1) + self.params.A + log_probs[t+1] + beta[t+1]

            log_xi -= alpha[-1].logsumexp(dim=0)
            xi_vec.append(log_xi)

        return xi_vec

    def _compute_posteriors(self, X:Observations) -> Tuple[List[torch.Tensor],...]:
        """Execute the forward-backward algorithm and compute the log-Gamma and log-Xi variables."""
        log_alpha = self._forward(X)
        log_beta = self._backward(X)
        log_gamma = self._gamma(log_alpha,log_beta)
        log_xi = self._xi(X,log_alpha,log_beta)
        return log_gamma, log_xi
    
    def _accum_pi(self, log_gamma:List[torch.Tensor]) -> torch.Tensor:
        """Accumulate the statistics for the initial vector."""
        log_pi = torch.zeros(size=(self.n_states,),
                             dtype=torch.float64)

        for gamma in log_gamma:
            log_pi += gamma[0].exp()
        
        return log_normalize(log_pi.log(),0)

    def _accum_A(self, log_xi:List[torch.Tensor]) -> torch.Tensor:
        """Accumulate the statistics for the transition matrix."""
        log_A = torch.zeros(size=(self.n_states,self.n_states),
                             dtype=torch.float64)
        
        for xi in log_xi:
            log_A += xi.exp().sum(dim=0)

        return log_normalize(log_A.log(),1)
    
    def _update_model(self, X:Observations, theta:Optional[ContextualVariables]) -> float:
        """Compute the updated parameters for the model."""
        log_gamma,log_xi = self._compute_posteriors(X)
        gamma = [torch.exp(gamma) for gamma in log_gamma]

        new_params = nn.ParameterDict({
            'pi':nn.Parameter(self._accum_pi(log_gamma),requires_grad=False),
            'A':nn.Parameter(self._accum_A(log_xi),requires_grad=False)
        })
        new_params.update(self.estimate_emission_params(X.X,gamma,theta))
        self.params.update(new_params)

        return sum(self._compute_log_likelihood(X))
    
    def _viterbi(self, X:Observations) -> Sequence[torch.Tensor]:
        """Viterbi algorithm for decoding the most likely sequence of hidden states."""
        viterbi_path_list = []
        for seq_len,_,log_probs in sequence_generator(X):
            viterbi_path = torch.empty(size=(seq_len,), 
                                       dtype=torch.int32)
            
            viterbi_prob = torch.empty(size=(self.n_states, seq_len), 
                                       dtype=torch.float64)
            psi = viterbi_prob.clone()

            # Initialize t=1
            viterbi_prob[:,0] = self.params.pi + log_probs[0]
            for t in range(1,seq_len):
                trans_seq = viterbi_prob[:,t-1] + log_probs[t]
                trans_seq = self.params.A + trans_seq.reshape((-1, 1))
                viterbi_prob[:,t] = torch.max(trans_seq, dim=0).values
                psi[:,t] = torch.argmax(trans_seq, dim=0)

            # Backtrack the most likely sequence
            viterbi_path[-1] = torch.argmax(viterbi_prob[:,-1])
            for t in reversed(range(seq_len-1)):
                viterbi_path[t] = psi[viterbi_path[t+1],t+1]

            viterbi_path_list.append(viterbi_path)

        return viterbi_path_list
    
    def _map(self, X:Observations) -> List[torch.Tensor]:
        """Compute the most likely (MAP) sequence of indiviual hidden states."""
        map_paths = []
        gamma_vec,_ = self._compute_posteriors(X)
        for gamma in gamma_vec:
            map_paths.append(torch.argmax(gamma, dim=1))
        return map_paths

    def _compute_log_likelihood(self, X:Observations) -> List[float]:
        """Compute the log-likelihood of the given sequence."""
        scores = []
        for alpha in self._forward(X):
            scores.append(alpha[-1].logsumexp(dim=0).item())

        return scores
