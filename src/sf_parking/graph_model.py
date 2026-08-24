"""Optional spatial-temporal neural forecaster.

This is a deliberately small GNN+GRU implementation using plain PyTorch and
an edge list; PyTorch Geometric is not required. It is a later-stage model,
not the first production baseline. Training still requires enough historical
snapshots and a trustworthy target from the occupancy layer.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class GraphTensorShape:
    timesteps: int
    nodes: int
    features: int


def _require_torch():
    try:
        import torch
        import torch.nn as nn
    except ImportError as exc:
        raise RuntimeError(
            "PyTorch is optional; install requirements-gnn.txt to use the GNN forecaster"
        ) from exc
    return torch, nn


def build_edge_index(edges: list[tuple[int, int]], *, device=None):
    """Create a two-row torch edge index from integer graph edges."""
    torch, _ = _require_torch()
    if not edges:
        return torch.empty((2, 0), dtype=torch.long, device=device)
    return torch.tensor(edges, dtype=torch.long, device=device).t().contiguous()


class SpatialTemporalGNN:
    """Graph message passing followed by a GRU over time.

    Input shape: [batch, time, nodes, features].
    Output shape: [batch, nodes] for the next-slot occupancy probability.
    """

    def __init__(
        self,
        *,
        node_features: int,
        hidden: int = 64,
    ) -> None:
        torch, nn = _require_torch()

        class Net(nn.Module):
            def __init__(self):
                super().__init__()
                self.node_in = nn.Linear(node_features, hidden)
                self.msg = nn.Linear(hidden, hidden)
                self.gru = nn.GRU(hidden, hidden, batch_first=True)
                self.out = nn.Linear(hidden, 1)

            def graph_step(self, x, edge_index):
                # x: [batch, nodes, hidden]
                src, dst = edge_index
                messages = x[:, src, :]
                agg = x.new_zeros(x.shape)
                agg.index_add_(1, dst, messages)
                degree = x.new_zeros((x.shape[1],))
                degree.index_add_(0, dst, x.new_ones((dst.shape[0],)))
                degree = degree.clamp_min(1).view(1, -1, 1)
                return x + torch.tanh(self.msg(agg / degree))

            def forward(self, x, edge_index):
                # Flatten batch/time for node projection, then run graph steps.
                b, t, n, f = x.shape
                h = torch.relu(self.node_in(x))
                graph_seq = []
                for ti in range(t):
                    graph_seq.append(self.graph_step(h[:, ti], edge_index))
                h = torch.stack(graph_seq, dim=1)  # [b,t,n,h]
                h = h.transpose(1, 2).contiguous().view(b * n, t, -1)
                out, _ = self.gru(h)
                last = out[:, -1]
                return torch.sigmoid(self.out(last)).view(b, n)

        self._net = Net()

    @property
    def module(self):
        return self._net

    def predict(self, x, edge_index):
        self._net.eval()
        return self._net(x, edge_index)
