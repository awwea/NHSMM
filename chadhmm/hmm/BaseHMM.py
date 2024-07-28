from abc import ABC, abstractmethod
from typing import Optional, List, Tuple, Literal, Any
import torch
import torch.nn as nn
from torch.distributions import Categorical, Distribution

from chadhmm.utilities import utils, constraints, SeedGenerator, ConvergenceHandler # type: ignore


class BaseHMM(nn.Module,ABC):
    """
    Base Abstract Class for HMM
    ----------
    Base Class of Hidden Markov Models (HMM) class that provides a foundation for building specific HMM models.
    """
    __slots__ = 'n_states','params'

    def __init__(self,
                 n_states:int,
                 transitions:constraints.Transitions,
                 alpha:float,
                 seed:Optional[int]):

        super().__init__()
        self.n_states = n_states
        self.alpha = alpha
        self._seed_gen = SeedGenerator(seed)
        self._A_type = transitions
        self._params = self.sample_model_params()
        
    @property
    def seed(self):
        self._seed_gen.seed

    @property
    def A(self) -> torch.Tensor:
        return self._params.A.logits
    
    @A.setter
    def A(self,logits:torch.Tensor):
        assert (o:=self.A.shape) == (f:=logits.shape), ValueError(f'Expected shape {o} but got {f}') 
        assert torch.allclose(logits.logsumexp(1),torch.ones(o)), ValueError(f'Probs do not sum to 1')
        assert constraints.is_valid_A(logits,self._A_type), ValueError(f'Transition Matrix is not satisfying the constraints given by its type {self._A_type}')
        self._params.A.logits = logits 

    @property
    def pi(self) -> torch.Tensor:
        return self._params.pi.logits
    
    @pi.setter
    def pi(self,logits:torch.Tensor):
        assert (o:=self.pi.shape) == (f:=logits.shape), ValueError(f'Expected shape {o} but got {f}')
        assert torch.allclose(logits.logsumexp(0),torch.ones(o)), ValueError(f'Probs do not sum to 1')
        self._params.pi.logits = logits

    @property
    def pdf(self) -> Any:
        return self._params.emission_pdf
    
    @property 
    @abstractmethod
    def dof(self) -> int:
        """Returns the degrees of freedom of the model."""
        pass

    @abstractmethod
    def _estimate_emission_pdf(self, 
                               X:torch.Tensor, 
                               posterior:torch.Tensor, 
                               theta:Optional[utils.ContextualVariables]) -> Distribution:
        """Update the emission parameters where posterior is of shape (n_states,n_samples)"""
        pass
        
    @abstractmethod
    def sample_emission_pdf(self, X:Optional[torch.Tensor]=None) -> Distribution:
        """Sample the emission parameters."""
        pass

    def sample_model_params(self, X:Optional[torch.Tensor]=None) -> nn.ParameterDict:
        """Initialize the model parameters."""
        sampled_pi = torch.log(constraints.sample_probs(self.alpha,(self.n_states,)))
        sampled_A = torch.log(constraints.sample_A(self.alpha,self.n_states,self._A_type))

        return nn.ParameterDict({
            'pi': Categorical(logits=sampled_pi),
            'A': Categorical(logits=sampled_A),
            'emission_pdf': self.sample_emission_pdf(X)   
        })

    def map_emission(self, x:torch.Tensor) -> torch.Tensor:
        """Get emission probabilities for a given sequence of observations."""
        pdf_shape = self.pdf.batch_shape + self.pdf.event_shape
        b_size = torch.Size([torch.atleast_2d(x).size(0)]) + pdf_shape
        x_batched = x.unsqueeze(-len(pdf_shape)).expand(b_size)
        return self.pdf.log_prob(x_batched).squeeze()

    def sample(self, size:int) -> torch.Tensor:
        """Sample from underlying Markov chain"""
        sampled_path = torch.zeros(size,dtype=torch.int)
        sampled_path[0] = self._params.pi.sample([1])

        sample_chain = self._params.A.sample(torch.Size([size]))
        for idx in range(size-1):
            sampled_path[idx+1] = sample_chain[idx,sampled_path[idx]]

        return sampled_path
    
    def check_constraints(self, value:torch.Tensor) -> torch.Tensor:
        not_supported = value[torch.logical_not(self.pdf.support.check(value))].unique()
        events = self.pdf.event_shape
        event_dims = len(events)
        assert len(not_supported) == 0, ValueError(f'Values outside PDF support, got values: {not_supported.tolist()}')
        assert value.ndim == event_dims+1, ValueError(f'Expected number of dims differs from PDF constraints on event shape {events}')
        if event_dims > 0:
            assert value.shape[1:] == events, ValueError(f'PDF event shape differs, expected {events} but got {value.shape[1:]}')
        return value

    def to_observations(self, X:torch.Tensor, lengths:Optional[List[int]]=None) -> utils.Observations:
        """Convert a sequence of observations to an Observations object."""
        X_valid = self.check_constraints(X).double()
        n_samples = X_valid.size(0)
        if lengths is not None:
            assert (s:=sum(lengths)) == n_samples, ValueError(f'Lenghts do not sum to total number of samples provided {s} != {n_samples}')
            seq_lengths = lengths
        else:
            seq_lengths = [n_samples]
        
        tensor_list = list(torch.split(X_valid,seq_lengths))
        nested_tensor_probs = [self.map_emission(tens) for tens in tensor_list]

        return utils.Observations(
            tensor_list,
            nested_tensor_probs,
            seq_lengths
        )  
    
    def to_contextuals(self, theta:torch.Tensor, X:utils.Observations) -> utils.ContextualVariables:
        """Returns the parameters of the model."""
        if (n_dim:=theta.ndim) != 2:
            raise ValueError(f'Context must be 2-dimensional. Got {n_dim}.')
        elif theta.shape[1] not in (1, sum(X.lengths)):
            raise ValueError(f'Context must have shape (context_vars, 1) for time independent context or (context_vars,{sum(X.lengths)}) for time dependent. Got {theta.shape}.')
        else:
            n_context, n_observations = theta.shape
            time_dependent = n_observations == sum(X.lengths)
            adj_theta = torch.vstack((theta, torch.ones(size=(1,n_observations),
                                                        dtype=torch.float64)))
            if not time_dependent:
                adj_theta = adj_theta.expand(n_context+1, sum(X.lengths))

            context_matrix = torch.split(adj_theta,list(X.lengths),1)
            
            return utils.ContextualVariables(
                n_context, 
                context_matrix, 
                time_dependent
            ) 

    def fit(self,
            X:torch.Tensor,
            tol:float=0.01,
            max_iter:int=15,
            n_init:int=1,
            post_conv_iter:int=1,
            ignore_conv:bool=False,
            sample_B_from_X:bool=False,
            verbose:bool=True,
            plot_conv:bool=False,
            lengths:Optional[List[int]]=None,
            theta:Optional[torch.Tensor]=None):
        """Fit the model to the given sequence using the EM algorithm."""
        if sample_B_from_X:
            self._params.update({'emission_pdf': self.sample_emission_pdf(X)})

        X_valid = self.to_observations(X,lengths)
        valid_theta = self.to_contextuals(theta,X_valid) if theta is not None else None

        self.conv = ConvergenceHandler(
            tol=tol,
            max_iter=max_iter,
            n_init=n_init,
            post_conv_iter=post_conv_iter,
            verbose=verbose
        )

        for rank in range(n_init):
            if rank > 0:
                self._params.update(self.sample_model_params(X))
            
            self.conv.push_pull(self._compute_log_likelihood(X_valid).sum(),0,rank)
            for iter in range(1,self.conv.max_iter+1):
                new_params = self._estimate_model_params(X_valid,valid_theta)
                self._params.update(new_params)
                X_valid.log_probs = [self.map_emission(tens) for tens in X_valid.sequence]
                
                curr_log_like = self._compute_log_likelihood(X_valid).sum()
                converged = self.conv.push_pull(curr_log_like,iter,rank)
                if converged and verbose and not ignore_conv:
                    break
        
        if plot_conv:
            self.conv.plot_convergence()

        return self

    def predict(self, 
                X:torch.Tensor, 
                lengths:Optional[List[int]] = None,
                algorithm:Literal['map','viterbi'] = 'viterbi') -> List[torch.Tensor]:
        """Predict the most likely sequences of hidden states"""
        X_valid = self.to_observations(X,lengths)
        if algorithm == 'map':
            decoded_path = self._map(X_valid)
        elif algorithm == 'viterbi':
            decoded_path = self._viterbi(X_valid)
        else:
            raise ValueError(f'Unknown decoder algorithm {algorithm}')
        
        return decoded_path

    def score(self, 
              X:torch.Tensor,
              lengths:Optional[List[int]]=None,
              by_sample:bool=True) -> torch.Tensor:
        """Compute the joint log-likelihood"""
        X_valid = self.to_observations(X,lengths)
        log_likelihoods = self._compute_log_likelihood(X_valid)

        if by_sample:
            return log_likelihoods
        else:
            return log_likelihoods.sum(0,keepdim=True)

    def ic(self,
           X:torch.Tensor,
           lengths:Optional[List[int]] = None,
           by_sample:bool=True,
           criterion:constraints.InformCriteria = constraints.InformCriteria.AIC) -> torch.Tensor:
        """Calculates the information criteria for a given model."""
        log_likelihood = self.score(X,lengths,by_sample)
        information_criteria = constraints.compute_information_criteria(X.shape[0],log_likelihood,self.dof,criterion)
        return information_criteria
    
    def _forward(self, X:utils.Observations) -> List[torch.Tensor]:
        """Forward pass of the forward-backward algorithm."""
        alpha_vec:List[torch.Tensor] = []
        for seq_len,log_probs in zip(X.lengths,X.log_probs):
            log_alpha = torch.zeros(
                size=(seq_len,self.n_states), 
                dtype=torch.float64
            )
            
            log_alpha[0] = self.pi + log_probs[0]
            for t in range(seq_len-1):
                log_alpha[t+1] = log_probs[t+1] + torch.logsumexp(self.A + log_alpha[t].reshape(-1,1),dim=0)

            alpha_vec.append(log_alpha)
 
        return alpha_vec
    
    def _backward(self, X:utils.Observations) -> List[torch.Tensor]:
        """Backward pass of the forward-backward algorithm."""
        beta_vec:List[torch.Tensor] = []
        for seq_len,log_probs in zip(X.lengths,X.log_probs): 
            log_beta = torch.zeros(
                size=(seq_len,self.n_states),
                dtype=torch.float64
            )
            
            for t in reversed(range(seq_len-1)):
                log_beta[t] = torch.logsumexp(self.A + log_probs[t+1] + log_beta[t+1],dim=1)
            
            beta_vec.append(log_beta)

        return beta_vec

    def _compute_posteriors(self, X:utils.Observations) -> Tuple[List[torch.Tensor],...]:
        """Execute the forward-backward algorithm and compute the log-Gamma and log-Xi variables."""
        gamma_vec:List[torch.Tensor] = []
        xi_vec:List[torch.Tensor] = []
        
        log_alpha_vec = self._forward(X)
        log_beta_vec = self._backward(X)

        for log_alpha,log_beta,log_probs in zip(log_alpha_vec,log_beta_vec,X.log_probs):
            trans_alpha = self.A.unsqueeze(0) + log_alpha[:-1].unsqueeze(-1)
            probs_beta = log_probs[1:] + log_beta[1:]
            
            xi_vec.append(constraints.log_normalize(trans_alpha + probs_beta.unsqueeze(1),dim=(1,2)))
            gamma_vec.append(constraints.log_normalize(log_alpha + log_beta,dim=1))

        return gamma_vec, xi_vec
    
    def _estimate_model_params(self, X:utils.Observations, theta:Optional[utils.ContextualVariables]) -> nn.ParameterDict:
        """Compute the updated parameters for the model."""
        log_gamma,log_xi = self._compute_posteriors(X)

        new_pi = constraints.log_normalize(torch.stack([tens[0] for tens in log_gamma],1).logsumexp(1),0)
        new_A = constraints.log_normalize(torch.cat(log_xi).logsumexp(0))
        new_pdf = self._estimate_emission_pdf(
            X=torch.cat(X.sequence),
            posterior=torch.cat(log_gamma).exp(),
            theta=theta
        )
    
        return nn.ParameterDict({
            'pi': Categorical(logits=new_pi),
            'A': Categorical(logits=new_A),
            'emission_pdf': new_pdf
        })
    
    def _viterbi(self, X:utils.Observations) -> List[torch.Tensor]:
        """Viterbi algorithm for decoding the most likely sequence of hidden states."""
        viterbi_vec = []
        for seq_len,log_probs in zip(X.lengths,X.log_probs):
            viterbi_path = torch.empty(
                size=(seq_len,), 
                dtype=torch.int64
            )

            viterbi_prob = torch.empty(
                size=(seq_len,self.n_states), 
                dtype=torch.float64
            )
            
            psi = viterbi_prob.clone()

            viterbi_prob[0] = log_probs[0] + self.pi
            for t in range(1,seq_len):
                trans_seq = self.A + (viterbi_prob[t-1] + log_probs[t]).reshape(-1, 1)
                viterbi_prob[t] = torch.max(trans_seq,dim=0).values
                psi[t] = torch.argmax(trans_seq,dim=0)

            viterbi_path[-1] = torch.argmax(viterbi_prob[-1])
            for t in reversed(range(seq_len-1)):
                viterbi_path[t] = psi[t+1,viterbi_path[t+1]]

            viterbi_vec.append(viterbi_path)

        return viterbi_vec
    
    def _map(self, X:utils.Observations) -> List[torch.Tensor]:
        """Compute the most likely (MAP) sequence of indiviual hidden states."""
        gamma,_ = self._compute_posteriors(X)
        map_paths = [gamma.argmax(1) for gamma in gamma]
        return map_paths

    def _compute_log_likelihood(self, X:utils.Observations) -> torch.Tensor:
        """Compute the log-likelihood of the given sequence."""
        log_alpha_vec = self._forward(X)
        concated_fwd = torch.stack([log_alpha[-1] for log_alpha in log_alpha_vec],1)
        scores = concated_fwd.logsumexp(0)
        return scores
