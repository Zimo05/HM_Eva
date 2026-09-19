import torch
from torch import nn
import torch.nn.functional as F
from typing import List, Optional


class NodeEmbedding(nn.Module):
    def __init__(self, d_model: int, hidden_dim: int = None):
        super().__init__()
        self.d_model = d_model
        self.hidden_dim = hidden_dim or d_model
        self.score_mlp = nn.Sequential(
            nn.Linear(d_model, self.hidden_dim),
            nn.Tanh(),
            nn.Linear(self.hidden_dim, 1)
        )

        self.out_proj = nn.Linear(d_model, d_model)

    def _to_tensor(self, seq_list: List[torch.Tensor], device=None) -> torch.Tensor:
        xs = []
        for x in seq_list:
            if not isinstance(x, torch.Tensor):
                x = torch.tensor(x, dtype=torch.float32)
            xs.append(x)

        X = torch.stack(xs, dim=0)  # [n_seq, d_model]
        if device is not None:
            X = X.to(device)
        return X

    def attention_pool(self, X: torch.Tensor) -> torch.Tensor:
        scores = self.score_mlp(X).squeeze(-1)       # [n_seq]
        alpha = F.softmax(scores, dim=0)             # [n_seq]
        pooled = torch.sum(alpha.unsqueeze(-1) * X, dim=0)  # [d_model]
        return pooled

    def attention_pool_indexed(
        self,
        X: torch.Tensor,
        node_index: torch.Tensor,
        num_nodes: int,
    ) -> torch.Tensor:
        """Attention-pool all nodes with a flattened segmented operation.

        Parameters
        ----------
        X : Tensor [E, d_model]
            Sequence embeddings gathered for every valid node membership.
        node_index : LongTensor [E]
            Destination node for each row of ``X``.
        num_nodes : int
            Total number of nodes, including nodes with no memberships.

        Returns
        -------
        Tensor [N, d_model]
            Projected pooled embeddings.  Nodes with no selected sequences are
            exactly zero, matching the pipeline's historical loop behavior.
        """
        if X.ndim != 2 or X.shape[-1] != self.d_model:
            raise ValueError(
                "X must have shape [num_memberships, d_model], "
                f"got {tuple(X.shape)}"
            )
        if node_index.ndim != 1 or node_index.shape[0] != X.shape[0]:
            raise ValueError(
                "node_index must have one entry per membership, "
                f"got {tuple(node_index.shape)} for X {tuple(X.shape)}"
            )
        if num_nodes < 0:
            raise ValueError(f"num_nodes must be non-negative, got {num_nodes}")
        if X.shape[0] == 0:
            return X.new_zeros((num_nodes, self.d_model))

        node_index = node_index.to(device=X.device, dtype=torch.long)
        scores = self.score_mlp(X).squeeze(-1)       # [E]
        max_score = scores.new_full((num_nodes,), -torch.inf)
        max_score.scatter_reduce_(
            0, node_index, scores, reduce="amax", include_self=True
        )
        exp_score = torch.exp(scores - max_score[node_index])
        denominator = scores.new_zeros(num_nodes)
        denominator.scatter_add_(0, node_index, exp_score)
        alpha = exp_score / denominator[node_index]

        pooled = X.new_zeros((num_nodes, self.d_model))
        pooled.index_add_(0, node_index, alpha.unsqueeze(-1) * X)
        node_emb = self.out_proj(pooled)
        return torch.where(denominator.unsqueeze(-1) > 0, node_emb, 0.0)

    def mean_pool(self, X: torch.Tensor) -> torch.Tensor:
        return X.mean(dim=0)

    def forward(self, node) -> torch.Tensor:
        if len(node.val) == 0:
            raise ValueError(f"Node {node.node_id} has empty val.")

        device = next(self.parameters()).device
        X = self._to_tensor(node.val, device=device)   # [n_seq, d_model]
        pooled = self.attention_pool(X)
        node_emb = self.out_proj(pooled)
        return node_emb


class NodeInputFusion(nn.Module):
    """Fuse the node input  ``h_i^0 = [u_i ; s_i ; g_i]``  into a ``d_model`` vector.

    ``u_i`` : pooled THP sequence embedding              [N, d_model]
    ``s_i`` : structural numeric features                [N, struct_dim]
              + a learnable parent-id embedding (looked up from ``parent_idx``)
    ``g_i`` : Hawkes / split statistics                  [N, hawkes_dim]

    The concatenation is projected back to ``d_model`` so the rest of the
    pipeline (StructuralTreeBlock / CrossAttention) is unchanged.

    Parameters
    ----------
    d_model : int
        Sequence-embedding / output dimension.
    struct_dim : int
        Dimension of the structural numeric features ``s_i`` (excl. parent emb).
    hawkes_dim : int
        Dimension of the Hawkes/split features ``g_i``.
    num_nodes : int
        Number of tree nodes (parent embedding has ``num_nodes + 1`` rows; the
        extra row is the "no parent / root" slot).
    parent_emb_dim : int
        Dimension of the learnable parent-id embedding.
    """

    def __init__(
        self,
        d_model: int,
        struct_dim: int,
        hawkes_dim: int,
        num_nodes: int,
        parent_emb_dim: int = 8,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.d_model = d_model
        self.parent_emb = nn.Embedding(num_nodes + 1, parent_emb_dim)

        in_dim = d_model + struct_dim + parent_emb_dim + hawkes_dim
        self.fuse = nn.Sequential(
            nn.Linear(in_dim, 2 * d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(2 * d_model, d_model),
        )
        self.ln = nn.LayerNorm(d_model)

    def forward(
        self,
        u: torch.Tensor,            # [N, d_model]  pooled sequence info (u_i)
        struct_feat: torch.Tensor,  # [N, struct_dim]
        parent_idx: torch.Tensor,   # [N] long
        hawkes_feat: torch.Tensor,  # [N, hawkes_dim]
    ) -> torch.Tensor:
        parent_e = self.parent_emb(parent_idx)            # [N, parent_emb_dim]
        x = torch.cat([u, struct_feat, parent_e, hawkes_feat], dim=-1)
        return self.ln(self.fuse(x))                      # [N, d_model]
    
# Example usage (guarded so imports don't break):
# if __name__ == "__main__":
#     node_embedder = NodeEmbedding(d_model=512).to("cuda")
#     root_emb = node_embedder(root)
#     print(root_emb.shape)  # torch.Size([512])
