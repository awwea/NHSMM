import torch
import torch.nn as nn

class HMM(nn.Module):
    def __init__(self, num_states, num_observation_symbols):
        super(HMM, self).__init__()

        self.num_states = num_states
        self.num_observation_symbols = num_observation_symbols

        # Transition matrix
        self.A = nn.Parameter(torch.rand(num_states, num_states),requires_grad=False)
        self.A.data = self.A.data / self.A.data.sum(dim=1, keepdim=True)  # Normalize rows

        # Emission matrix
        self.B = nn.Parameter(torch.rand(num_states, num_observation_symbols),requires_grad=False)
        self.B.data = self.B.data / self.B.data.sum(dim=1, keepdim=True)  # Normalize rows

        # Initial state distribution
        self.pi = nn.Parameter(torch.rand(num_states),requires_grad=False)
        self.pi.data = self.pi.data / self.pi.data.sum()  # Normalize

    def forward(self, observations):
        """
        Forward algorithm for inference.
        """
        T = len(observations)
        alpha = torch.zeros(T, self.num_states)

        # Initialization
        alpha[0, :] = self.pi * self.B[:, observations[0]]

        # Recursion
        for t in range(1, T):
            alpha[t, :] = torch.matmul(alpha[t-1, :], self.A) * self.B[:, observations[t]]

        return alpha

    def train(self, observations, max_iters=100, epsilon=1e-6):
        """
        Baum-Welch algorithm for training.
        """
        T = len(observations)
        num_observation_symbols = max(max(observations) + 1, self.num_observation_symbols)

        for _ in range(max_iters):
            # E-step
            alpha = self.forward(observations)
            beta = torch.zeros_like(alpha)

            # Backward pass
            for t in reversed(range(T-1)):
                beta[t, :] = torch.matmul(self.A, (self.B[:, observations[t+1]] * beta[t+1, :]))
                beta[t, :] /= beta[t, :].sum()

            # M-step
            gamma = alpha * beta
            gamma /= gamma.sum(dim=1, keepdim=True)

            xi = torch.zeros(T-1, self.num_states, self.num_states)
            for t in range(T-1):
                xi[t, :, :] = alpha[t, :].view(-1, 1) * self.A * self.B[:, observations[t+1]].view(1, -1) * beta[t+1, :].view(1, -1)
                xi[t, :, :] /= xi[t, :, :].sum()

            # Update parameters
            self.pi.data = gamma[0, :]
            self.A.data = xi.sum(dim=0) / gamma[:-1, :].sum(dim=0, keepdim=True)
            self.B.data = torch.zeros(self.num_states, num_observation_symbols)
            for k in range(num_observation_symbols):
                mask = (observations == k)
                self.B[:, k] = gamma[mask, :].sum(dim=0) / gamma.sum(dim=0)

            # Check for convergence
            if torch.max(torch.abs(gamma.sum(dim=1) - 1)) < epsilon:
                break

if __name__ == '__main__':
    # Example usage
    num_states = 2
    num_observation_symbols = 3
    model = HMM(num_states, num_observation_symbols)

    # Generate synthetic data
    torch.manual_seed(42)
    observations = torch.randint(0, num_observation_symbols, (100,))

    # Train the model
    model.train(observations)

    # Perform inference
    test_observations = torch.randint(0, num_observation_symbols, (50,))
    alpha = model.forward(test_observations)
    print("Inference result:")
    print(alpha[-1, :])
    print(model.A.data)
