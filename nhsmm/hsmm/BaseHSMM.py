# nhsmm/hsmm/BaseHSMM.py
from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Optional, List, Tuple, Any, Literal, Dict
import torch
import torch.nn as nn
from torch.distributions import Distribution, Categorical

from nhsmm.utilities import utils, constraints, SeedGenerator, ConvergenceHandler

DTYPE = torch.float64


class BaseHSMM(nn.Module, ABC):
    """
    Base HSMM core (probabilistic). 
    - Implements EM (Baum-Welch for semi-Markov), forward/backward, Viterbi, scoring, IC.
    - Keeps pi/A/D logits registered as buffers for persistence.
    - Delegates emission parameterization to subclasses via `sample_emission_pdf` and `_estimate_emission_pdf`.
    - Provides small, well-defined hooks for neural/contextual integration:
        - attach_encoder / encode_observations
        - _contextual_emission_pdf, _contextual_transition_matrix, _contextual_duration_pdf
    """

    def __init__(self,
                 n_states: int,
                 max_duration: int,
                 alpha: float = 1.0,
                 seed: Optional[int] = None):
        super().__init__()
        self.n_states = int(n_states)
        self.max_duration = int(max_duration)
        self.alpha = float(alpha)
        self._seed_gen = SeedGenerator(seed)

        # container for emitted distribution and other non-buffer parameters
        self._params: Dict[str, Any] = {}

        # optional external encoder (neural) for contextualization
        self.encoder: Optional[nn.Module] = None

        # initialize & register pi/A/D logits as buffers (log-space)
        self._init_buffers()

        # initialize emission pdf by calling subclass hook (may be based on no data)
        self._params['emission_pdf'] = self.sample_emission_pdf(None)

    # ----------------------
    # Persistence buffers
    # ----------------------
    def _init_buffers(self):
        sampled_pi = torch.log(constraints.sample_probs(self.alpha, (self.n_states,))).to(dtype=DTYPE)
        sampled_A = torch.log(constraints.sample_A(self.alpha, self.n_states, constraints.Transitions.SEMI)).to(dtype=DTYPE)
        sampled_D = torch.log(constraints.sample_probs(self.alpha, (self.n_states, self.max_duration))).to(dtype=DTYPE)

        # register buffers so state_dict() contains them
        self.register_buffer("_pi_logits", sampled_pi)
        self.register_buffer("_A_logits", sampled_A)
        self.register_buffer("_D_logits", sampled_D)

    # ----------------------
    # Properties (log-space tensors)
    # ----------------------
    @property
    def seed(self) -> Optional[int]:
        return self._seed_gen.seed

    @property
    def pi(self) -> torch.Tensor:
        return self._pi_logits

    @pi.setter
    def pi(self, logits: torch.Tensor):
        logits = logits.to(dtype=DTYPE)
        if logits.shape != (self.n_states,):
            raise ValueError(f"pi logits must have shape ({self.n_states},) but got {tuple(logits.shape)}")
        if not torch.allclose(logits.logsumexp(0), torch.tensor(0.0, dtype=DTYPE), atol=1e-8):
            raise ValueError("pi logits must normalize (logsumexp==0)")
        self._pi_logits.copy_(logits)

    @property
    def A(self) -> torch.Tensor:
        return self._A_logits

    @A.setter
    def A(self, logits: torch.Tensor):
        logits = logits.to(dtype=DTYPE)
        if logits.shape != (self.n_states, self.n_states):
            raise ValueError(f"A logits must have shape ({self.n_states},{self.n_states})")
        if not torch.allclose(logits.logsumexp(1), torch.zeros(self.n_states, dtype=DTYPE), atol=1e-8):
            raise ValueError("Rows of A logits must normalize (logsumexp==0)")
        if not constraints.is_valid_A(logits, constraints.Transitions.SEMI):
            raise ValueError("A logits do not satisfy SEMI constraints")
        self._A_logits.copy_(logits)

    @property
    def D(self) -> torch.Tensor:
        return self._D_logits

    @D.setter
    def D(self, logits: torch.Tensor):
        logits = logits.to(dtype=DTYPE)
        if logits.shape != (self.n_states, self.max_duration):
            raise ValueError(f"D logits must have shape ({self.n_states},{self.max_duration})")
        if not torch.allclose(logits.logsumexp(1), torch.zeros(self.n_states, dtype=DTYPE), atol=1e-8):
            raise ValueError("Rows of D logits must normalize (logsumexp==0)")
        self._D_logits.copy_(logits)

    @property
    def pdf(self) -> Any:
        """Emission distribution (torch.distributions.Distribution). Managed by subclass via hooks."""
        return self._params.get('emission_pdf')

    # ----------------------
    # Subclass API (abstract)
    # ----------------------
    @property
    @abstractmethod
    def dof(self) -> int:
        """Degrees of freedom (required for IC computations)."""
        raise NotImplementedError

    @abstractmethod
    def sample_emission_pdf(self, X: Optional[torch.Tensor] = None) -> Distribution:
        """
        Create and return an initial emission Distribution.
        Called at construction and optionally with data X when sample_B_from_X=True.
        """
        raise NotImplementedError

    @abstractmethod
    def _estimate_emission_pdf(self,
                               X: torch.Tensor,
                               posterior: torch.Tensor,
                               theta: Optional[utils.ContextualVariables]) -> Distribution:
        """
        Given X (n_samples, *event_shape) and posterior (n_states, n_samples),
        return an updated emission Distribution. Subclass should update any
        registered buffers for persistence.
        """
        raise NotImplementedError

    # ----------------------
    # Neural/context hooks (override as needed)
    # ----------------------
    def attach_encoder(self, encoder: nn.Module):
        """
        Attach a neural encoder module (e.g. CNN+LSTM). Encoder must accept a single sequence
        shaped (1, T, F) or (T, F) and return either:
          - a tensor of shape (hidden_dim,) representing sequence-level context, or
          - a tensor of shape (hidden_dim, T) or (hidden_dim, 1) compatible with to_contextuals().
        The BaseHSMM will transform encoder output into `theta` used by contextual hooks.
        """
        self.encoder = encoder

    def encode_observations(self, X: torch.Tensor) -> Optional[torch.Tensor]:
        """
        Run attached encoder (if present) on X and return a 2D tensor `theta`.
        Default behaviour: return None if no encoder attached.
        The returned tensor should follow to_contextuals() expectations: shape (n_context, 1) for time-independent context.
        """
        if self.encoder is None:
            return None

        # Ensure input shape (T, F) -> encoder expects (1, T, F)
        if X.ndim == 2:
            inp = X.unsqueeze(0)
        elif X.ndim == 3:
            # batch not supported in base; take first batch
            inp = X[0:1]
        else:
            raise ValueError("Unsupported input shape for encoder; expected (T,F) or (B,T,F).")

        out = self.encoder(inp)  # allow encoder to return (seq, state) or tensor
        if isinstance(out, tuple):
            out = out[0]
        # out expected (B, T, H) or (B, H) or (B,1,H). normalize to (H,1)
        out = out.detach()
        if out.ndim == 3:
            # take last time: (B, T, H) -> (H,)
            vec = out[0, -1, :].to(dtype=DTYPE)
        elif out.ndim == 2:
            vec = out[0].to(dtype=DTYPE)
        elif out.ndim == 1:
            vec = out.to(dtype=DTYPE)
        else:
            raise ValueError("Encoder output dims not supported.")

        return vec.unsqueeze(-1)  # shape (H, 1) matches to_contextuals time-independent form

    def _contextual_emission_pdf(self, X: utils.Observations, theta: Optional[utils.ContextualVariables]) -> Distribution:
        """
        Hook: return a (possibly) context-adapted emission pdf.
        Default: return stored emission pdf unchanged.
        Neural subclasses override to map theta -> new emission Distribution.
        """
        return self._params.get('emission_pdf')

    def _contextual_transition_matrix(self, theta: Optional[utils.ContextualVariables]) -> torch.Tensor:
        """Hook: return A logits (log-space). Default: current buffer."""
        return self._A_logits

    def _contextual_duration_pdf(self, theta: Optional[utils.ContextualVariables]) -> torch.Tensor:
        """Hook: return D logits (log-space). Default: current buffer."""
        return self._D_logits

    # ----------------------
    # Emission mapping & validation
    # ----------------------
    def map_emission(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute emission log-probabilities per state for sequence x.
        Returns tensor shape (T, n_states).
        """
        pdf = self.pdf
        if pdf is None:
            raise RuntimeError("Emission PDF not initialized; subclass must implement sample_emission_pdf")

        # Ensure x is (T, *event_shape)
        # For scalar event_shape, distribution.log_prob expects shape (T,)
        if len(pdf.event_shape) == 0:
            # scalar emission: pass (T,) and rely on batch_shape broadcasting
            return pdf.log_prob(x)
        else:
            # vector/matrix event shape: check shape
            if x.shape[1:] != pdf.event_shape:
                raise ValueError(f"Input event shape {x.shape[1:]} != pdf.event_shape {pdf.event_shape}")
            # unsqueeze state axis so distribution broadcasting applies: pdf.batch_shape == (n_states,)
            # pdf.log_prob( x.unsqueeze(1) ) => (T, n_states)
            return pdf.log_prob(x.unsqueeze(1))

    def check_constraints(self, value: torch.Tensor) -> torch.Tensor:
        """Validate observations against pdf support and event_shape."""
        pdf = self.pdf
        if pdf is None:
            raise RuntimeError("Emission PDF not initialized.")
        not_supported = value[torch.logical_not(pdf.support.check(value))].unique()
        events = pdf.event_shape
        event_dims = len(events)
        if len(not_supported) != 0:
            raise ValueError(f'Values outside PDF support, got values: {not_supported.tolist()}')
        if value.ndim != event_dims + 1:
            raise ValueError(f'Expected dims != PDF event shape {events}')
        if event_dims > 0 and value.shape[1:] != events:
            raise ValueError(f'PDF event shape differs, expected {events} but got {value.shape[1:]}')
        return value

    def to_observations(self, X: torch.Tensor, lengths: Optional[List[int]] = None) -> utils.Observations:
        """
        Convert X -> utils.Observations which contains:
          - sequence: list of tensors (per sequence)
          - log_probs: list of log-prob tensors (per sequence, shape (T, n_states))
          - lengths: list of lengths
        """
        X_valid = self.check_constraints(X).to(dtype=DTYPE)
        n_samples = X_valid.size(0)
        if lengths is not None:
            if sum(lengths) != n_samples:
                raise ValueError("Lengths do not sum to total samples.")
            seq_lengths = lengths
        else:
            seq_lengths = [n_samples]

        tensor_list = list(torch.split(X_valid, seq_lengths))
        tensor_probs = [self.map_emission(tens) for tens in tensor_list]
        return utils.Observations(sequence=tensor_list, log_probs=tensor_probs, lengths=seq_lengths)

    # ----------------------
    # Parameter sampling & EM loop
    # ----------------------
    def sample_model_params(self, X: Optional[torch.Tensor] = None) -> Dict[str, Any]:
        sampled_pi = torch.log(constraints.sample_probs(self.alpha, (self.n_states,))).to(dtype=DTYPE)
        sampled_A = torch.log(constraints.sample_A(self.alpha, self.n_states, constraints.Transitions.SEMI)).to(dtype=DTYPE)
        sampled_D = torch.log(constraints.sample_probs(self.alpha, (self.n_states, self.max_duration))).to(dtype=DTYPE)
        sampled_pdf = self.sample_emission_pdf(X)
        return {'pi': sampled_pi, 'A': sampled_A, 'D': sampled_D, 'emission_pdf': sampled_pdf}

    def fit(self,
            X: torch.Tensor,
            tol: float = 1e-4,
            max_iter: int = 15,
            n_init: int = 1,
            post_conv_iter: int = 1,
            ignore_conv: bool = False,
            sample_B_from_X: bool = False,
            verbose: bool = True,
            plot_conv: bool = False,
            lengths: Optional[List[int]] = None,
            theta: Optional[torch.Tensor] = None):
        """
        Fit model using EM. If `theta` is None and an encoder is attached, encoder output will be used.
        `theta` expected to be a 2D tensor compatible with to_contextuals() when present.
        """
        # if user wants to initialize emission from X
        if sample_B_from_X:
            self._params['emission_pdf'] = self.sample_emission_pdf(X)

        # If encoder attached and no external theta provided, encode to get theta
        if theta is None and self.encoder is not None:
            theta = self.encode_observations(X)

        X_valid = self.to_observations(X, lengths)
        valid_theta = self.to_contextuals(theta, X_valid) if theta is not None else None

        self.conv = ConvergenceHandler(
            tol=tol,
            max_iter=max_iter,
            n_init=n_init,
            post_conv_iter=post_conv_iter,
            verbose=verbose
        )

        best_state = None
        best_score = -float('inf')

        for rank in range(n_init):
            if rank > 0:
                sampled = self.sample_model_params(X)
                self._pi_logits.copy_(sampled['pi'])
                self._A_logits.copy_(sampled['A'])
                self._D_logits.copy_(sampled['D'])
                self._params['emission_pdf'] = sampled['emission_pdf']

            self.conv.push_pull(self._compute_log_likelihood(X_valid).sum(), 0, rank)

            for iter_idx in range(1, self.conv.max_iter + 1):
                # let contextual hooks adapt emission/transition/duration before E-step
                self._params['emission_pdf'] = self._contextual_emission_pdf(X_valid, valid_theta)
                A_logits = self._contextual_transition_matrix(valid_theta)
                D_logits = self._contextual_duration_pdf(valid_theta)
                # copy into buffers if hooks returned tensors
                if A_logits is not None:
                    self._A_logits.copy_(A_logits.to(dtype=DTYPE))
                if D_logits is not None:
                    self._D_logits.copy_(D_logits.to(dtype=DTYPE))

                new_params = self._estimate_model_params(X_valid, valid_theta)

                # update emission and logits from returned dict
                if 'emission_pdf' in new_params:
                    self._params['emission_pdf'] = new_params['emission_pdf']
                if 'pi' in new_params:
                    self._pi_logits.copy_(new_params['pi'].to(dtype=DTYPE))
                if 'A' in new_params:
                    self._A_logits.copy_(new_params['A'].to(dtype=DTYPE))
                if 'D' in new_params:
                    self._D_logits.copy_(new_params['D'].to(dtype=DTYPE))

                # recompute emission log-probs (now emission may have changed)
                X_valid.log_probs = [self.map_emission(seq) for seq in X_valid.sequence]

                curr_log_like = self._compute_log_likelihood(X_valid).sum()
                converged = self.conv.push_pull(curr_log_like, iter_idx, rank)

                if converged and verbose and not ignore_conv:
                    break

            run_score = float(self._compute_log_likelihood(X_valid).sum().item())
            if run_score > best_score:
                best_score = run_score
                best_state = {
                    'pi': self._pi_logits.clone(),
                    'A': self._A_logits.clone(),
                    'D': self._D_logits.clone(),
                    'emission_pdf': self._params.get('emission_pdf')
                }

        # restore best run
        if best_state is not None:
            self._pi_logits.copy_(best_state['pi'])
            self._A_logits.copy_(best_state['A'])
            self._D_logits.copy_(best_state['D'])
            self._params['emission_pdf'] = best_state['emission_pdf']

        if plot_conv and hasattr(self, 'conv'):
            self.conv.plot_convergence()

        return self

    # ----------------------
    # Predict / Score / IC
    # ----------------------
    def predict(self,
                X: torch.Tensor,
                lengths: Optional[List[int]] = None,
                algorithm: Literal['map', 'viterbi'] = 'viterbi') -> List[torch.Tensor]:
        X_valid = self.to_observations(X, lengths)
        if algorithm == 'map':
            return self._map(X_valid)
        elif algorithm == 'viterbi':
            return self._viterbi(X_valid)
        else:
            raise ValueError(f"Unknown decode algorithm {algorithm}")

    def score(self,
              X: torch.Tensor,
              lengths: Optional[List[int]] = None,
              by_sample: bool = True) -> torch.Tensor:
        X_valid = self.to_observations(X, lengths)
        log_likelihoods = self._compute_log_likelihood(X_valid)
        return log_likelihoods if by_sample else log_likelihoods.sum(0, keepdim=True)

    def ic(self,
           X: torch.Tensor,
           criterion: constraints.InformCriteria = constraints.InformCriteria.AIC,
           lengths: Optional[List[int]] = None,
           by_sample: bool = True) -> torch.Tensor:
        log_likelihood = self.score(X, lengths, by_sample)
        return constraints.compute_information_criteria(X.shape[0], log_likelihood, self.dof, criterion)

    # ----------------------
    # Forward / Backward / Posteriors
    # ----------------------
    def _forward(self, X: utils.Observations) -> List[torch.Tensor]:
        alpha_vec: List[torch.Tensor] = []
        pi, A, D = self.pi, self.A, self.D
        neg_inf = float('-inf')

        for seq_probs, seq_len in zip(X.log_probs, X.lengths):
            log_alpha = torch.full((seq_len, self.n_states, self.max_duration), neg_inf, dtype=DTYPE)

            # t=0 initialization
            log_alpha[0] = pi.unsqueeze(-1) + seq_probs[0].unsqueeze(-1) + D

            for t in range(1, seq_len):
                prev_alpha = log_alpha[t - 1]

                # shift durations
                shifted = torch.cat([prev_alpha[:, 1:], torch.full((self.n_states, 1), neg_inf, dtype=DTYPE)], dim=1)

                # new durations start after transition
                trans = torch.logsumexp(prev_alpha[:, 0].unsqueeze(1) + A, dim=0)
                log_alpha[t] = torch.logaddexp(shifted + seq_probs[t].unsqueeze(-1), D + trans.unsqueeze(-1))

            alpha_vec.append(log_alpha)

        return alpha_vec

    def _backward(self, X: utils.Observations) -> List[torch.Tensor]:
        """
        Vectorized backward pass for semi-Markov HMM.
        Returns a list of log-beta tensors per sequence: shape (T, n_states, max_duration)
        """
        beta_vec: List[torch.Tensor] = []
        A, D = self.A, self.D
        neg_inf = float('-inf')

        for seq_probs, seq_len in zip(X.log_probs, X.lengths):
            device = seq_probs.device
            log_beta = torch.full((seq_len, self.n_states, self.max_duration), neg_inf, dtype=DTYPE, device=device)

            # Last time-step initialization: beta[T-1, :, :] = 0 for all durations
            log_beta[-1, :, :] = 0.0

            # Precompute cumulative sums of emissions for durations
            # cumsum_emit[t] = sum_{tau=t}^{t+d-1} log p(x_tau | state)
            cumsum_emit = torch.vstack((torch.zeros((1, self.n_states), dtype=DTYPE, device=device),
                                        torch.cumsum(seq_probs, dim=0)))  # shape (T+1, n_states)

            for t in reversed(range(seq_len - 1)):
                max_d = min(self.max_duration, seq_len - t)

                # Compute contributions for new durations starting at t
                durations = torch.arange(1, max_d + 1, dtype=torch.int64, device=device)
                emit_sums = cumsum_emit[t + durations] - cumsum_emit[t]  # (durations, n_states)
                emit_sums = emit_sums.T  # (n_states, durations)

                dur_lp = D[:, :max_d]  # (n_states, durations)

                beta_next = log_beta[t + durations - 1, :, 0]  # (durations, n_states)
                beta_next = beta_next.T  # (n_states, durations)

                # Combine emissions, duration probabilities, and beta
                contrib = emit_sums + dur_lp + beta_next  # (n_states, durations)

                # Logsumexp over durations to get beta[t, :, 0]
                log_beta[t, :, 0] = torch.logsumexp(contrib, dim=1)

                # Shift ongoing durations
                if self.max_duration > 1:
                    log_beta[t, :, 1:] = log_beta[t + 1, :, :-1] + seq_probs[t + 1].unsqueeze(-1)

            beta_vec.append(log_beta)

        return beta_vec

    def _compute_posteriors(self, X: utils.Observations) -> Tuple[List[torch.Tensor], ...]:
        """
        Compute gamma (state marginals), xi (pairwise transitions), and eta (duration marginals)
        for a batch of sequences.
        """
        alpha_list = self._forward(X)
        beta_list = self._backward(X)
        gamma_vec, xi_vec, eta_vec = [], [], []

        for seq_probs, seq_len, alpha, beta in zip(X.log_probs, X.lengths, alpha_list, beta_list):
            # Gamma: marginal over states at each time
            gamma = torch.logsumexp(alpha + beta, dim=2)
            gamma = constraints.log_normalize(gamma, 1)
            gamma_vec.append(gamma.exp())

            # Eta: marginal over durations
            # log_eta[t, s, d] ~ alpha[t, s, d] + beta[t, s, d]
            log_eta = alpha + beta
            eta_vec.append(constraints.log_normalize(log_eta).exp())

            # Xi: marginal over transitions (s -> s') at each t
            if seq_len > 1:
                trans_alpha = alpha[:-1, :, 0].unsqueeze(1) + self.A.unsqueeze(0)  # (T-1, n_states, n_states)
                dur_beta = beta[1:]  # (T-1, n_states, max_duration)
                dur_beta_sum = torch.logsumexp(dur_beta + seq_probs[1:].unsqueeze(-1), dim=2)  # (T-1, n_states)
                log_xi = trans_alpha + dur_beta_sum.unsqueeze(1)
                xi_vec.append(constraints.log_normalize(log_xi, (1, 2)).exp())
            else:
                xi_vec.append(torch.empty((0, self.n_states, self.n_states), dtype=DTYPE))

        return gamma_vec, xi_vec, eta_vec

    def _estimate_model_params(self, X: utils.Observations, theta: Optional[utils.ContextualVariables]) -> Dict[str, Any]:
        gamma_list, xi_list, eta_list = self._compute_posteriors(X)

        # ----------------------
        # Update pi
        # ----------------------
        # stack first time-step of gamma from all sequences: shape (n_states, n_sequences)
        pi_stack = torch.stack([g[0] for g in gamma_list], dim=1).to(dtype=DTYPE)
        new_pi = constraints.log_normalize(torch.log(pi_stack.sum(dim=1)), 0)

        # ----------------------
        # Update A (transition matrix)
        # ----------------------
        # concatenate xi along time dimension for all sequences: (total_time, n_states, n_states)
        xi_cat = torch.cat([x for x in xi_list], dim=0)
        # logsumexp over time dimension
        new_A = constraints.log_normalize(torch.logsumexp(xi_cat, dim=0), 1)

        # ----------------------
        # Update D (duration distributions)
        # ----------------------
        # concatenate eta along time dimension for all sequences: (total_time, n_states, max_duration)
        eta_cat = torch.cat([e for e in eta_list], dim=0)
        new_D = constraints.log_normalize(torch.logsumexp(eta_cat, dim=0), 1)

        # ----------------------
        # Update emission pdf
        # ----------------------
        all_X = torch.cat(X.sequence, dim=0)
        all_gamma = torch.cat([g for g in gamma_list], dim=0)  # shape (total_time, n_states)
        new_pdf = self._estimate_emission_pdf(all_X, all_gamma, theta)

        return {'pi': new_pi, 'A': new_A, 'D': new_D, 'emission_pdf': new_pdf}

    # ----------------------
    # Viterbi (semi-Markov)
    # ----------------------
    def _viterbi(self, X: utils.Observations) -> List[torch.Tensor]:
        paths: List[torch.Tensor] = []
        K = self.n_states
        Dmax = self.max_duration
        A = self.A
        pi = self.pi
        D = self.D
        neg_inf = -1e300

        for seq_probs, seq_len in zip(X.log_probs, X.lengths):
            T = seq_len
            if T == 0:
                paths.append(torch.empty((0,), dtype=torch.int64))
                continue

            seq_probs = seq_probs.to(dtype=DTYPE)
            # DP tables
            V = torch.full((T, K), neg_inf, dtype=DTYPE)
            best_prev = torch.full((T, K), -1, dtype=torch.int64)
            best_dur = torch.zeros((T, K), dtype=torch.int64)

            # Precompute cumulative sums for emissions
            cumsum_emit = torch.vstack((torch.zeros((1, K), dtype=DTYPE), torch.cumsum(seq_probs, dim=0)))

            # Compute all possible durations vectorized
            for t in range(T):
                max_d = min(Dmax, t + 1)
                durations = torch.arange(1, max_d + 1, dtype=torch.int64)
                starts = t - durations + 1
                emit_sums = cumsum_emit[t + 1] - cumsum_emit[starts]  # (durations, K)

                # Expand dimensions for broadcasting
                emit_sums = emit_sums.T  # (K, durations)
                dur_lp = D[:, :max_d]  # (K, durations)

                if t == 0:
                    scores = pi.unsqueeze(1) + dur_lp + emit_sums  # (K, durations)
                    prev_idx = torch.full_like(scores, -1, dtype=torch.int64)
                else:
                    V_prev = V[t - durations]  # (durations, K)
                    # Compute score for each previous state -> current state -> duration
                    scores_plus_trans = V_prev.unsqueeze(2) + A.unsqueeze(0)  # (durations, K_prev, K)
                    scores_max, argmax_prev = scores_plus_trans.max(1)  # (durations, K)
                    scores = scores_max.T + dur_lp + emit_sums  # (K, durations)
                    prev_idx = argmax_prev.T  # (K, durations)

                # Pick best duration per state
                best_score_dur, best_d_idx = scores.max(dim=1)
                V[t] = best_score_dur
                best_dur[t] = durations[best_d_idx]
                best_prev[t] = prev_idx[torch.arange(K), best_d_idx]

            # Backtracking
            t = T - 1
            cur_state = int(V[T - 1].argmax().item())
            decoded_segments = []

            while t >= 0:
                d = int(best_dur[t, cur_state].item())
                start = t - d + 1
                decoded_segments.append((start, t, cur_state))
                prev_state = int(best_prev[t, cur_state].item())
                t = start - 1
                cur_state = prev_state if prev_state >= 0 else cur_state

            decoded_segments.reverse()
            seq_path = torch.empty((sum(e - s + 1 for s, e, _ in decoded_segments),), dtype=torch.int64)
            idx = 0
            for s, e, st in decoded_segments:
                L = e - s + 1
                seq_path[idx: idx + L] = st
                idx += L

            paths.append(seq_path)

        return paths

    def _map(self, X: utils.Observations) -> List[torch.Tensor]:
        gamma, _, _ = self._compute_posteriors(X)
        return [tens.argmax(1) for tens in gamma]

    def _compute_log_likelihood(self, X: utils.Observations) -> torch.Tensor:
        log_alpha_vec = self._forward(X)
        concated_fwd = torch.stack([log_alpha[-1] for log_alpha in log_alpha_vec], 1)
        return concated_fwd.logsumexp(0)
