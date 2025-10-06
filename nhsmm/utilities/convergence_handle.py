import torch
import matplotlib.pyplot as plt
import json
from typing import Callable, List, Optional


class ConvergenceHandler:
    """
    Robust convergence monitor for HMM/HSMM training.
    Supports both EM-based and neural training loops (e.g., CNN/LSTM HSMMs).

    Parameters
    ----------
    max_iter : int
        Maximum number of iterations per initialization.
    n_init : int
        Number of model initializations.
    tol : float
        Convergence threshold on log-likelihood delta.
    post_conv_iter : int
        Minimum number of iterations to continue after convergence.
    verbose : bool, default=True
        Print iteration status.
    callbacks : list of callables, optional
        Each callback is called as:
            fn(handler, iter: int, rank: int, score: float, delta: float, converged: bool)
    early_stop : bool, default=True
        If True, sets `handler.stop_training` when convergence is reached.
    """

    def __init__(
        self,
        max_iter: int,
        n_init: int,
        tol: float,
        post_conv_iter: int,
        verbose: bool = True,
        callbacks: Optional[List[Callable]] = None,
        early_stop: bool = True,
    ):
        self.max_iter = int(max_iter)
        self.n_init = int(n_init)
        self.tol = float(tol)
        self.post_conv_iter = int(post_conv_iter)
        self.verbose = verbose
        self.early_stop = early_stop

        # logs
        self.score = torch.full((max_iter + 1, n_init), float("nan"), dtype=torch.float64)
        self.delta = torch.full_like(self.score, float("nan"))
        self.is_converged = False
        self.stop_training = False
        self.callbacks = callbacks or []

    def __repr__(self):
        return (
            f"ConvergenceHandler("
            f"max_iter={self.max_iter}, n_init={self.n_init}, "
            f"tol={self.tol}, post_conv_iter={self.post_conv_iter}, "
            f"early_stop={self.early_stop}, verbose={self.verbose})"
        )

    # ------------------------- Core API -------------------------

    def push_pull(self, new_score: torch.Tensor, iter: int, rank: int) -> bool:
        """Push a new score and immediately check convergence."""
        self.push(new_score, iter, rank)
        return self.check_converged(iter, rank)

    def push(self, new_score: torch.Tensor, iter: int, rank: int):
        """Store a new log-likelihood score and compute delta."""
        score_val = float(new_score.detach().item()) if torch.is_tensor(new_score) else float(new_score)
        self.score[iter, rank] = score_val

        if iter > 0 and not torch.isnan(self.score[iter - 1, rank]):
            self.delta[iter, rank] = score_val - self.score[iter - 1, rank]
        else:
            self.delta[iter, rank] = float("nan")

    def check_converged(self, iter: int, rank: int) -> bool:
        """Determine convergence, trigger callbacks, and control early stopping."""
        if iter < self.post_conv_iter:
            self.is_converged = False
        else:
            recent = self.delta[max(1, iter - self.post_conv_iter + 1) : iter + 1, rank]
            self.is_converged = torch.all(torch.abs(recent) < self.tol).item()

        # Logging
        if self.verbose:
            score = self.score[iter, rank].item()
            delta = self.delta[iter, rank].item() if not torch.isnan(self.delta[iter, rank]) else float("nan")
            msg = (
                f"[Run {rank+1}] Iter {iter:03d} | Score: {score:.4f} | Δ: {delta:.6f} | "
                f"{'✔️ Converged' if self.is_converged else ''}"
            )
            print(msg)

        # Callbacks
        self._trigger_callbacks(iter, rank)

        # Early stop flag
        if self.is_converged and self.early_stop:
            self.stop_training = True

        return self.is_converged

    # ------------------------- Callback API -------------------------

    def _trigger_callbacks(self, iter: int, rank: int):
        """Invoke registered callbacks."""
        score = float(self.score[iter, rank])
        delta = float(self.delta[iter, rank]) if not torch.isnan(self.delta[iter, rank]) else float("nan")
        for fn in self.callbacks:
            try:
                fn(self, iter, rank, score, delta, self.is_converged)
            except Exception as e:
                if self.verbose:
                    print(f"[Callback Error] {fn.__name__}: {e}")

    def register_callback(self, fn: Callable):
        """Dynamically register a new callback."""
        self.callbacks.append(fn)

    # ------------------------- Visualization -------------------------

    def plot_convergence(self, show: bool = True, savepath: Optional[str] = None):
        """Plot convergence for each run."""
        plt.style.use("ggplot")
        fig, ax = plt.subplots(figsize=(10, 6))
        iters = torch.arange(self.max_iter + 1)

        for r in range(self.score.shape[1]):
            ax.plot(iters, self.score[:, r].cpu(), marker="o", linewidth=1.8, label=f"Run #{r+1}")

        ax.set_title("Log-Likelihood Convergence")
        ax.set_xlabel("Iteration")
        ax.set_ylabel("Log-Likelihood")
        ax.legend(loc="lower right")

        if savepath:
            plt.savefig(savepath, bbox_inches="tight")
        if show:
            plt.show()
        plt.close(fig)

    # ------------------------- Logging / Export -------------------------

    def export_log(self, path: str):
        """Export score and delta logs to JSON."""
        data = {
            "max_iter": self.max_iter,
            "n_init": self.n_init,
            "tol": self.tol,
            "scores": self.score.cpu().numpy().tolist(),
            "delta": self.delta.cpu().numpy().tolist(),
        }
        with open(path, "w") as f:
            json.dump(data, f, indent=2)

    def reset(self):
        """Reset state for reuse."""
        self.score.fill_(float("nan"))
        self.delta.fill_(float("nan"))
        self.is_converged = False
        self.stop_training = False
