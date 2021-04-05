from typing import List

import numpy as np
import torch
import torch.nn.functional as F

from torch import nn, optim, Tensor

from config import config


class EmbeddingModel(nn.Module):
    def __init__(self, shape, num_outputs, n_hidden=256):
        super().__init__()
        self.shape = torch.Size(shape)
        self.num_outputs = num_outputs

        self.embedding = nn.Sequential(
            nn.Flatten(-len(self.shape), -1),
            nn.Linear(self.shape.numel(), n_hidden, bias=True),
            nn.ReLU(),
            nn.Linear(n_hidden, n_hidden, bias=True),
            nn.ReLU(),
        )
        self.embedding.shape = torch.Size((n_hidden,))

        self.classifier = nn.Sequential(
            nn.Linear(n_hidden * 2, num_outputs, bias=True),
            nn.LogSoftmax(dim=-1),
        )

        self.criterion = nn.NLLLoss(reduction='mean')

        self.optimizer = optim.Adam(self.parameters(), lr=1e-5)

    def forward(self, x1, x2):
        return self.classifier(torch.cat([
            self.embedding(x1),
            self.embedding(x2),
        ], dim=-1))

    def train_model(self, batch):
        states, next_states, actions = map(
            torch.stack, (batch.state, batch.next_state, batch.action))

        batch_size, sequence_length, *shape = states.shape
        assert sequence_length == config.sequence_length
        assert torch.Size(shape) == self.shape

        # last 5 in sequence (see paper appendix A)
        states, next_states, actions = states[:, -5:], next_states[:, -5:], actions[:, -5:]

        self.optimizer.zero_grad()
        net_out = self.forward(states, next_states)
        loss = self.criterion(net_out.transpose(-1, -2), actions)

        loss.backward()
        self.optimizer.step()
        return float(loss)


def compute_intrinsic_reward(
    episodic_memory: List,
    current_c_state: Tensor,
    k=10,
    kernel_cluster_distance=0.008,
    kernel_epsilon=0.0001,
    c=0.001,
    sm=8,
) -> float:
    state_dist = [(c_state, torch.dist(c_state, current_c_state)) for c_state in episodic_memory]
    state_dist.sort(key=lambda x: x[1])
    state_dist = state_dist[:k]
    dist = [d[1].item() for d in state_dist]
    dist = np.array(dist)

    # TODO: moving average
    dist = dist / np.mean(dist)

    dist = np.max(dist - kernel_cluster_distance, 0)
    kernel = kernel_epsilon / (dist + kernel_epsilon)
    s = np.sqrt(np.sum(kernel)) + c

    if np.isnan(s) or s > sm:
        return 0
    return 1 / s
