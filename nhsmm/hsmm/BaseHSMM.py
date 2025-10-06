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
        device = next(self.parameters(), torch.tensor(0.)).device
        sampled_pi = torch.log(constraints.sample_probs(self.alpha, (self.n_states,))).to(device=device, dtype=DTYPE)
        sampled_A = torch.log(constraints.sample_A(self.alpha, self.n_states, constraints.Transitions.SEMI)).to(device=device, dtype=DTYPE)
        sampled_D = torch.log(constraints.sample_probs(self.alpha, (self.n_states, self.max_duration))).to(device=device, dtype=DTYPE)

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
        logits = logits.to(device=self._pi_logits.device, dtype=DTYPE)
        if logits.shape != (self.n_states,):
            raise ValueError(f"pi logits must have shape ({self.n_states},) but got {tuple(logits.shape)}")

        norm_val = logits.logsumexp(0)
        if not torch.allclose(norm_val, torch.tensor(0.0, dtype=DTYPE, device=logits.device), atol=1e-8):
            raise ValueError(f"pi logits must normalize (logsumexp==0); got {norm_val.item():.3e}")

        self._pi_logits.copy_(logits)

    @property
    def A(self) -> torch.Tensor:
        return self._A_logits

    @A.setter
    def A(self, logits: torch.Tensor):
        logits = logits.to(device=self._A_logits.device, dtype=DTYPE)
        if logits.shape != (self.n_states, self.n_states):
            raise ValueError(f"A logits must have shape ({self.n_states},{self.n_states})")

        row_norm = logits.logsumexp(1)
        if not torch.allclose(row_norm, torch.zeros_like(row_norm), atol=1e-8):
            raise ValueError(f"Rows of A logits must normalize (logsumexp==0); got {row_norm}")

        if not constraints.is_valid_A(logits, constraints.Transitions.SEMI):
            raise ValueError("A logits do not satisfy SEMI transition constraints")

        self._A_logits.copy_(logits)

    @property
    def D(self) -> torch.Tensor:
        return self._D_logits

    @D.setter
    def D(self, logits: torch.Tensor):
        logits = logits.to(device=self._D_logits.device, dtype=DTYPE)
        if logits.shape != (self.n_states, self.max_duration):
            raise ValueError(f"D logits must have shape ({self.n_states},{self.max_duration})")

        row_norm = logits.logsumexp(1)
        if not torch.allclose(row_norm, torch.zeros_like(row_norm), atol=1e-8):
            raise ValueError(f"Rows of D logits must normalize (logsumexp==0); got {row_norm}")

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
        Encode observations into a context vector θ.
        Returns a tensor of shape (H, 1) suitable for time-independent contextual adaptation.
        If no encoder is attached, returns None.
        """
        if self.encoder is None:
            return None

        # Ensure device alignment
        device = next(self.encoder.parameters()).device
        if X.ndim == 2:
            inp = X.unsqueeze(0).to(device)
        elif X.ndim == 3:
            inp = X[0:1].to(device)  # Base version uses only first batch
        else:
            raise ValueError("Unsupported input shape for encoder; expected (T,F) or (B,T,F).")

        out = self.encoder(inp)
        if isinstance(out, tuple):  # Handle (output, hidden) encoders
            out = out[0]
        out = out.detach()

        # Normalize output to a single context vector
        if out.ndim == 3:      # (B, T, H)
            vec = out[0, -1, :].to(dtype=DTYPE)
        elif out.ndim == 2:    # (B, H)
            vec = out[0].to(dtype=DTYPE)
        elif out.ndim == 1:    # (H,)
            vec = out.to(dtype=DTYPE)
        else:
            raise ValueError(f"Unsupported encoder output shape {out.shape}")

        return vec.unsqueeze(-1)  # (H, 1)

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
        Compute per-state emission log-probabilities for sequence x.
        Returns a tensor of shape (T, n_states).
        """
        pdf = self.pdf
        if pdf is None:
            raise RuntimeError("Emission PDF not initialized; subclass must implement `sample_emission_pdf()`.")

        # Ensure input matches expected event shape
        event_shape = pdf.event_shape
        if event_shape and x.shape[1:] != event_shape:
            raise ValueError(f"Input event shape {x.shape[1:]} does not match PDF event shape {event_shape}.")

        # Always expand a state dimension for broadcasting
        # Works for both scalar and structured emissions
        return pdf.log_prob(x.unsqueeze(1))

    def check_constraints(self, value: torch.Tensor) -> torch.Tensor:
        """Validate observations against the emission PDF support and event shape."""
        pdf = self.pdf
        if pdf is None:
            raise RuntimeError("Emission PDF not initialized.")

        # Support validation
        support_mask = pdf.support.check(value)
        if not torch.all(support_mask):
            bad_vals = value[~support_mask].unique()
            raise ValueError(f"Values outside PDF support detected: {bad_vals.tolist()}")

        # Shape validation
        event_shape = pdf.event_shape
        expected_ndim = len(event_shape) + 1  # batch + event dims
        if value.ndim != expected_ndim:
            raise ValueError(f"Expected {expected_ndim}D input (batch + event), got {value.ndim}D.")
        if event_shape and value.shape[1:] != event_shape:
            raise ValueError(f"PDF event shape mismatch: expected {event_shape}, got {value.shape[1:]}.")

        return value

    def to_observations(self, X: torch.Tensor, lengths: Optional[List[int]] = None) -> utils.Observations:
        """
        Convert tensor X into a utils.Observations object containing:
          - sequence: list of tensors (split by lengths)
          - log_probs: list of log-probability tensors (shape (T_i, n_states) per sequence)
          - lengths: list of sequence lengths

        Args:
            X: Tensor of shape (N, feature_dim) or similar.
            lengths: Optional list of per-sequence lengths summing to N.

        Returns:
            utils.Observations: Container with split sequences, emission log-probs, and lengths.
        """
        device = next(self.parameters()).device
        X_valid = self.check_constraints(X).to(dtype=DTYPE, device=device)
        n_samples = X_valid.size(0)

        if lengths is not None:
            if sum(lengths) != n_samples:
                raise ValueError(f"Sum of lengths ({sum(lengths)}) does not match total samples ({n_samples}).")
            seq_lengths = lengths
        else:
            seq_lengths = [n_samples]

        tensor_list = list(torch.split(X_valid, seq_lengths))
        tensor_probs = [self.map_emission(seq) for seq in tensor_list]

        return utils.Observations(sequence=tensor_list, log_probs=tensor_probs, lengths=seq_lengths)

    # ----------------------
    # Parameter sampling & EM loop
    # ----------------------
    def sample_model_params(self, X: Optional[torch.Tensor] = None) -> Dict[str, Any]:
        """
        Randomly sample initial model parameters (π, A, D, emission_pdf)
        in log-space for EM initialization.
        """
        device = next(self.parameters()).device
        α = self.alpha

        sampled_pi = torch.log(constraints.sample_probs(α, (self.n_states,))).to(dtype=DTYPE, device=device)
        sampled_A = torch.log(constraints.sample_A(α, self.n_states, constraints.Transitions.SEMI)).to(dtype=DTYPE, device=device)
        sampled_D = torch.log(constraints.sample_probs(α, (self.n_states, self.max_duration))).to(dtype=DTYPE, device=device)

        sampled_pdf = self.sample_emission_pdf(X)

        return {
            "pi": sampled_pi,
            "A": sampled_A,
            "D": sampled_D,
            "emission_pdf": sampled_pdf
        }

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
        Fit the model using EM optimization.

        Args:
            X: Observation tensor of shape (T, F) or batched sequence.
            tol: Convergence tolerance for log-likelihood.
            max_iter: Maximum EM iterations per initialization.
            n_init: Number of random initializations.
            post_conv_iter: Iterations to run after convergence.
            ignore_conv: If True, ignores convergence and runs full max_iter.
            sample_B_from_X: Initialize emission distribution from data.
            verbose: Print progress.
            plot_conv: Plot convergence diagnostics if available.
            lengths: Optional list of sequence lengths.
            theta: Optional contextual encoding tensor.

        Returns:
            self: The fitted model instance.
        """
        device = next(self.parameters()).device

        # Optionally initialize emission distribution from data
        if sample_B_from_X:
            self._params['emission_pdf'] = self.sample_emission_pdf(X)

        # Encode observations if model has encoder and no external context
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

        best_state, best_score = None, -float("inf")

        for run_idx in range(n_init):
            if verbose:
                print(f"\n=== Run {run_idx + 1}/{n_init} ===")

            # Random reinitialization for multi-start
            if run_idx > 0:
                sampled = self.sample_model_params(X)
                self._pi_logits.copy_(sampled['pi'].to(dtype=DTYPE, device=device))
                self._A_logits.copy_(sampled['A'].to(dtype=DTYPE, device=device))
                self._D_logits.copy_(sampled['D'].to(dtype=DTYPE, device=device))
                self._params['emission_pdf'] = sampled['emission_pdf']

            # Initialize convergence tracker with base likelihood
            base_ll = self._compute_log_likelihood(X_valid).sum()
            self.conv.push_pull(base_ll, 0, run_idx)

            for it in range(1, max_iter + 1):
                # Contextual adaptation (encoder-conditioned parameters)
                self._params['emission_pdf'] = self._contextual_emission_pdf(X_valid, valid_theta)
                A_logits = self._contextual_transition_matrix(valid_theta)
                D_logits = self._contextual_duration_pdf(valid_theta)

                if A_logits is not None:
                    self._A_logits.copy_(A_logits.to(dtype=DTYPE, device=device))
                if D_logits is not None:
                    self._D_logits.copy_(D_logits.to(dtype=DTYPE, device=device))

                # Expectation–Maximization step
                new_params = self._estimate_model_params(X_valid, valid_theta)

                if 'emission_pdf' in new_params:
                    self._params['emission_pdf'] = new_params['emission_pdf']
                if 'pi' in new_params:
                    self._pi_logits.copy_(new_params['pi'].to(dtype=DTYPE, device=device))
                if 'A' in new_params:
                    self._A_logits.copy_(new_params['A'].to(dtype=DTYPE, device=device))
                if 'D' in new_params:
                    self._D_logits.copy_(new_params['D'].to(dtype=DTYPE, device=device))

                # Refresh emission log probabilities
                X_valid.log_probs = [self.map_emission(seq) for seq in X_valid.sequence]

                curr_ll = self._compute_log_likelihood(X_valid).sum()
                converged = self.conv.push_pull(curr_ll, it, run_idx)

                if converged:
                    if verbose:
                        print(f"[Run {run_idx + 1}] Converged at iteration {it}.")
                    if not ignore_conv:
                        break

            run_score = float(self._compute_log_likelihood(X_valid).sum().item())
            if run_score > best_score:
                best_score = run_score
                best_state = {
                    'pi': self._pi_logits.clone(),
                    'A': self._A_logits.clone(),
                    'D': self._D_logits.clone(),
                    'emission_pdf': self._params['emission_pdf']
                }

        # Restore best model parameters
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
    def predict(
        self,
        X: torch.Tensor,
        lengths: Optional[List[int]] = None,
        algorithm: Literal["map", "viterbi"] = "viterbi"
    ) -> List[torch.Tensor]:
        """
        Decode the most likely hidden state sequences using either MAP or Viterbi algorithm.

        Args:
            X: Input tensor of shape (T, F) or (B, T, F).
            lengths: Optional list of sequence lengths (for batched inputs).
            algorithm: Decoding strategy — 'map' for posterior-based decoding,
                       or 'viterbi' for maximum-likelihood path.

        Returns:
            List[torch.Tensor]: One decoded state sequence per input.
        """
        X_valid = self.to_observations(X, lengths)

        decoders = {
            "map": self._map,
            "viterbi": self._viterbi
        }

        alg = algorithm.lower()
        if alg not in decoders:
            raise ValueError(
                f"Unknown decode algorithm '{algorithm}'. Expected one of {list(decoders.keys())}."
            )

        return decoders[alg](X_valid)

    def score(
        self,
        X: torch.Tensor,
        lengths: Optional[List[int]] = None,
        by_sample: bool = True
    ) -> torch.Tensor:
        """Compute log-likelihood(s) of input sequence(s)."""
        obs = self.to_observations(X, lengths)
        log_likelihoods = self._compute_log_likelihood(obs)
        return log_likelihoods if by_sample else log_likelihoods.sum(dim=0, keepdim=True)

    def ic(
        self,
        X: torch.Tensor,
        criterion: constraints.InformCriteria = constraints.InformCriteria.AIC,
        lengths: Optional[List[int]] = None,
        by_sample: bool = True
    ) -> torch.Tensor:
        log_likelihood = self.score(X, lengths, by_sample)
        n_obs = len(lengths) if lengths is not None else X.shape[0]
        return constraints.compute_information_criteria(n_obs, log_likelihood, self.dof, criterion)

    # ----------------------
    # Forward / Backward / Posteriors
    # ----------------------
    def _forward(self, X: utils.Observations) -> List[torch.Tensor]:
        """
        Log-domain forward recursion for explicit-duration HSMM.
        Returns: list of tensors (T, n_states, max_duration)
        """
        device = next(self.parameters()).device
        pi, A, D = self.pi, self.A, self.D
        neg_inf = -torch.inf
        alpha_vec: List[torch.Tensor] = []

        for seq_probs, seq_len in zip(X.log_probs, X.lengths):
            log_alpha = torch.full((seq_len, self.n_states, self.max_duration), neg_inf, dtype=DTYPE, device=device)

            # Precompute cumulative emission sums
            cumsum_emit = torch.vstack((torch.zeros((1, self.n_states), dtype=DTYPE, device=device),
                                        torch.cumsum(seq_probs, dim=0)))

            # t = 0 initialization
            max_d = min(self.max_duration, seq_len)
            durations = torch.arange(1, max_d + 1, device=device)
            emit_sums = (cumsum_emit[durations] - cumsum_emit[0]).T  # (n_states, durations)
            log_alpha[0, :, :max_d] = pi.unsqueeze(-1) + D[:, :max_d] + emit_sums

            # Recursion
            for t in range(1, seq_len):
                prev_alpha = log_alpha[t - 1]

                # Continuation of durations
                shifted = torch.cat([
                    prev_alpha[:, 1:],
                    torch.full((self.n_states, 1), neg_inf, dtype=DTYPE, device=device)
                ], dim=1)

                # Transition to new durations
                trans = torch.logsumexp(prev_alpha[:, 0].unsqueeze(1) + A, dim=0)
                log_alpha[t] = torch.logsumexp(
                    torch.stack([
                        shifted + seq_probs[t].unsqueeze(-1),
                        D + trans.unsqueeze(-1)
                    ]),
                    dim=0
                )

            alpha_vec.append(log_alpha)

        return alpha_vec

    def _backward(self, X: utils.Observations) -> List[torch.Tensor]:
        """
        Log-domain backward recursion for explicit-duration HSMM.
        Returns: list of tensors (T, n_states, max_duration)
        """
        beta_vec: List[torch.Tensor] = []
        A, D = self.A, self.D
        neg_inf = -torch.inf

        for seq_probs, seq_len in zip(X.log_probs, X.lengths):
            device = seq_probs.device
            log_beta = torch.full((seq_len, self.n_states, self.max_duration),
                                  neg_inf, dtype=DTYPE, device=device)
            log_beta[-1].fill_(0.)

            # Precompute cumulative emission sums
            cumsum_emit = torch.vstack((
                torch.zeros((1, self.n_states), dtype=DTYPE, device=device),
                torch.cumsum(seq_probs, dim=0)
            ))

            durations_all = torch.arange(1, self.max_duration + 1, device=device)

            for t in reversed(range(seq_len - 1)):
                max_d = min(self.max_duration, seq_len - t)
                durations = durations_all[:max_d]

                # Emission sums over durations [t, t+d)
                emit_sums = cumsum_emit[t + durations] - cumsum_emit[t]  # (durations, n_states)
                emit_sums = emit_sums.T  # (n_states, durations)

                dur_lp = D[:, :max_d]
                beta_next = log_beta[t + durations - 1, :, 0].T  # (n_states, durations)

                # Combine emissions, duration, and next beta
                contrib = emit_sums + dur_lp + beta_next
                log_beta[t, :, 0] = torch.logsumexp(contrib, dim=1)

                # Continue existing durations
                if self.max_duration > 1:
                    log_beta[t, :, 1:] = log_beta[t + 1, :, :-1] + seq_probs[t + 1].unsqueeze(-1)

            beta_vec.append(log_beta)

        return beta_vec

    def _compute_posteriors(self, X: utils.Observations) -> Tuple[List[torch.Tensor], ...]:
        """
        Compute posterior expectations:
          γ_t(s)  = state marginals
          ξ_t(s,s') = transition marginals
          η_t(s,d) = duration marginals
        Returns lists aligned with X.sequences.
        """
        alpha_list = self._forward(X)
        beta_list = self._backward(X)
        gamma_vec, xi_vec, eta_vec = [], [], []

        for seq_probs, seq_len, alpha, beta in zip(X.log_probs, X.lengths, alpha_list, beta_list):
            device = alpha.device

            # --- State marginals ---
            gamma = torch.logsumexp(alpha + beta, dim=2)
            gamma = constraints.log_normalize(gamma, dim=1).exp()
            gamma_vec.append(gamma)

            # --- Duration marginals ---
            log_eta = alpha + beta
            eta = constraints.log_normalize(log_eta, dim=(1, 2)).exp()
            eta_vec.append(eta)

            # --- Transition marginals ---
            if seq_len > 1:
                trans_alpha = alpha[:-1, :, 0].unsqueeze(1) + self.A.unsqueeze(0)  # (T-1, s, s')
                dur_beta_sum = torch.logsumexp(beta[1:] + self.D, dim=2)           # (T-1, s')
                log_xi = trans_alpha + dur_beta_sum.unsqueeze(1)
                xi = constraints.log_normalize(log_xi, dim=(1, 2)).exp()
                xi_vec.append(xi)
            else:
                xi_vec.append(torch.empty((0, self.n_states, self.n_states), dtype=DTYPE, device=device))

        return gamma_vec, xi_vec, eta_vec

    def _estimate_model_params(self, X: utils.Observations, theta: Optional[utils.ContextualVariables]) -> Dict[str, Any]:
        """
        M-step: Estimate updated parameters (π, A, D, emission_pdf) from posterior expectations.
        """
        gamma_list, xi_list, eta_list = self._compute_posteriors(X)
        device = next(self.parameters()).device

        # --- π (initial state probabilities)
        pi_stack = torch.stack([g[0] for g in gamma_list], dim=1).to(device)  # (n_states, n_sequences)
        new_pi = constraints.log_normalize(torch.log(pi_stack.sum(dim=1)), dim=0)

        # --- A (state transition matrix)
        xi_valid = [x for x in xi_list if x.numel() > 0]
        if len(xi_valid) > 0:
            xi_cat = torch.cat(xi_valid, dim=0)
            new_A = constraints.log_normalize(torch.logsumexp(xi_cat, dim=0), dim=1)
        else:
            new_A = self.A

        # --- D (duration distributions)
        eta_cat = torch.cat([e for e in eta_list], dim=0)
        new_D = constraints.log_normalize(torch.logsumexp(eta_cat, dim=0), dim=1)

        # --- Emission PDF
        all_X = torch.cat(X.sequence, dim=0).to(device)
        all_gamma = torch.cat([g for g in gamma_list], dim=0).to(device)
        new_pdf = self._estimate_emission_pdf(all_X, all_gamma, theta)

        return {'pi': new_pi, 'A': new_A, 'D': new_D, 'emission_pdf': new_pdf}

    # ----------------------
    # Viterbi (semi-Markov)
    # ----------------------
    @torch.no_grad()
    def _viterbi(self, X: utils.Observations) -> List[torch.Tensor]:
        """
        HSMM Viterbi decoding (MAP state sequence)
        """
        device = next(self.parameters()).device
        K, Dmax = self.n_states, self.max_duration
        A, pi, D = self.A.to(device), self.pi.to(device), self.D.to(device)
        neg_inf = -torch.inf
        paths = []

        for seq_probs, seq_len in zip(X.log_probs, X.lengths):
            T = seq_len
            if T == 0:
                paths.append(torch.empty((0,), dtype=torch.int64, device=device))
                continue

            seq_probs = seq_probs.to(dtype=DTYPE, device=device)
            V = torch.full((T, K), neg_inf, dtype=DTYPE, device=device)
            best_prev = torch.full((T, K), -1, dtype=torch.int64, device=device)
            best_dur = torch.zeros((T, K), dtype=torch.int64, device=device)

            cumsum_emit = torch.vstack((torch.zeros((1, K), dtype=DTYPE, device=device),
                                        torch.cumsum(seq_probs, dim=0)))

            for t in range(T):
                max_d = min(Dmax, t + 1)
                durations = torch.arange(1, max_d + 1, dtype=torch.int64, device=device)
                starts = t - durations + 1
                emit_sums = (cumsum_emit[t + 1] - cumsum_emit[starts]).T  # (K, durations)
                dur_lp = D[:, :max_d]

                if t == 0:
                    scores = pi.unsqueeze(1) + dur_lp + emit_sums
                    prev_idx = torch.full_like(scores, -1, dtype=torch.int64)
                else:
                    V_prev = V[t - durations]
                    scores_plus_trans = V_prev.unsqueeze(2) + A.unsqueeze(0)
                    scores_max, argmax_prev = scores_plus_trans.max(1)
                    scores = scores_max.T + dur_lp + emit_sums
                    prev_idx = argmax_prev.T

                best_score_dur, best_d_idx = scores.max(dim=1)
                V[t] = best_score_dur
                best_dur[t] = durations[best_d_idx]
                best_prev[t] = prev_idx[torch.arange(K), best_d_idx]

            # Backtrack
            t = T - 1
            cur_state = int(V[t].argmax().item())
            decoded_segments = []

            while t >= 0:
                d = int(best_dur[t, cur_state].item())
                start = t - d + 1
                decoded_segments.append((start, t, cur_state))
                prev_state = int(best_prev[t, cur_state].item())
                if prev_state < 0:
                    break
                t = start - 1
                cur_state = prev_state

            decoded_segments.reverse()
            seq_path = torch.cat([torch.full((e - s + 1,), st, dtype=torch.int64, device=device)
                                  for s, e, st in decoded_segments])
            paths.append(seq_path)

        return paths

    def _map(self, X: utils.Observations) -> List[torch.Tensor]:
        gamma, _, _ = self._compute_posteriors(X)
        return [tens.argmax(1) for tens in gamma]

    def _compute_log_likelihood(self, X: utils.Observations) -> torch.Tensor:
        log_alpha_vec = self._forward(X)
        return torch.stack([log_alpha[-1].logsumexp(0) for log_alpha in log_alpha_vec]).sum()
