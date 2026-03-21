"""ShipHGT: Heterogeneous Graph Transformer encoder for ship designs."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch_geometric.nn import HGTConv

from training.backends.hgt.convert import METADATA
from training.backends.hgt.vocab import WEAPON_TYPES

__all__ = ["ShipHGT", "SinusoidalPE"]

# Virtual node types that can be dropped during training as conditioning dropout.
_VIRTUAL_TYPES: tuple[str, ...] = (
    "cluster", "thermal", "hull_peri", "hull_int",
    "zone", "zone_rot", "weapon_grp", "ship_info",
)

# Simple virtual types whose input projection is a single scalar → hidden_dim.
_SIMPLE_VIRT_TYPES: tuple[str, ...] = (
    "cluster", "thermal", "hull_peri", "hull_int", "ship_info",
)


class SinusoidalPE(nn.Module):
    """2D sinusoidal positional encoding for (x, y) grid coordinates.

    Returns a ``[N, pe_dim]`` tensor; ``pe_dim`` must be divisible by 4
    (``pe_dim // 4`` frequencies are used per axis).
    """

    def __init__(self, pe_dim: int = 32) -> None:
        super().__init__()
        assert pe_dim % 4 == 0, "pe_dim must be divisible by 4"
        half = pe_dim // 4
        freqs = 1.0 / (10000.0 ** (torch.arange(half, dtype=torch.float) / half))
        self.register_buffer("freqs", freqs)

    def forward(self, coords: Tensor) -> Tensor:
        # coords: [N, 2]
        x = coords[:, 0:1] * self.freqs  # [N, half]
        y = coords[:, 1:2] * self.freqs  # [N, half]
        return torch.cat([x.sin(), x.cos(), y.sin(), y.cos()], dim=1)


class ShipHGT(nn.Module):
    """HGT encoder for heterogeneous ship graphs.

    Architecture
    ------------
    1. Per-node-type input projections map raw features to *hidden_dim*.
    2. Stacked :class:`~torch_geometric.nn.HGTConv` layers perform
       heterogeneous message passing.
    3. A linear head predicts masked part_id values for the pretraining
       objective.

    Parameters
    ----------
    vocab_size:
        Total vocabulary size including ``<unk>`` and ``<mask>`` tokens.
    hidden_dim:
        Width of all hidden representations.  Must be divisible by
        ``num_heads``.
    num_heads:
        Number of attention heads in each HGTConv layer.
    num_layers:
        Number of stacked HGTConv layers.
    dropout:
        Dropout probability applied after each projection and conv layer.
    pe_dim:
        Dimensionality of the sinusoidal positional encoding for part
        coordinates.  Must be divisible by 4.
    virtual_dropout_rate:
        Probability with which each virtual node type's features are zeroed
        during the training forward pass (conditioning dropout).
    """

    def __init__(
        self,
        vocab_size: int,
        *,
        hidden_dim: int = 128,
        num_heads: int = 4,
        num_layers: int = 3,
        dropout: float = 0.1,
        pe_dim: int = 32,
        virtual_dropout_rate: float = 0.3,
    ) -> None:
        super().__init__()
        assert hidden_dim % num_heads == 0, "hidden_dim must be divisible by num_heads"

        self.hidden_dim = hidden_dim
        self.virtual_dropout_rate = virtual_dropout_rate
        # Set of virtual node types zeroed by the last _apply_virtual_dropout call.
        # Always frozenset() during eval; reflects actual drops during training.
        self._last_dropped_virtual_types: frozenset[str] = frozenset()

        # --- Input projections: structural parts ---
        part_embed_dim = hidden_dim // 2  # 64 by default
        self.part_id_embed = nn.Embedding(vocab_size, part_embed_dim)
        self.rotation_embed = nn.Embedding(4, 16)
        self.loc_pe = SinusoidalPE(pe_dim)
        # part input: part_id_embed + rotation_embed + pe + [fp_cells, fp_w, fp_h, traversable, overclocked]
        part_in_dim = part_embed_dim + 16 + pe_dim + 5
        self.part_proj = nn.Linear(part_in_dim, hidden_dim)

        # --- Input projections: virtual nodes ---
        # cluster / thermal / hull_peri / hull_int / ship_info: 1 scalar → hidden_dim
        self.simple_virt_projs = nn.ModuleDict(
            {ntype: nn.Linear(1, hidden_dim) for ntype in _SIMPLE_VIRT_TYPES}
        )

        # zone / zone_rot: 1 scalar + zone_label_embed(16) → hidden_dim
        self.zone_label_embed = nn.Embedding(8, 16)
        self.zone_proj = nn.Linear(1 + 16, hidden_dim)
        self.zone_rot_proj = nn.Linear(1 + 16, hidden_dim)

        # weapon_grp: 1 scalar + weapon_type_embed(16) → hidden_dim
        self.weapon_type_embed = nn.Embedding(len(WEAPON_TYPES), 16)
        self.weapon_grp_proj = nn.Linear(1 + 16, hidden_dim)

        # --- HGTConv layers ---
        self.convs = nn.ModuleList(
            [HGTConv(hidden_dim, hidden_dim, METADATA, num_heads) for _ in range(num_layers)]
        )

        self.dropout = nn.Dropout(dropout)

        # --- Masked part prediction head ---
        # Predicts over num_classes (all valid part_ids + <unk>, excludes <mask>).
        self.part_pred_head = nn.Linear(hidden_dim, vocab_size - 1)

        # --- Auxiliary masked prediction heads ---
        # Binary overclocked status prediction for masked parts.
        self.overclock_pred_head = nn.Linear(hidden_dim, 1)
        # Edge existence prediction from concatenated endpoint embeddings.
        self.edge_pred_head = nn.Linear(hidden_dim * 2, 1)

    # ------------------------------------------------------------------
    # Input encoding
    # ------------------------------------------------------------------

    def _encode(self, data: "HeteroData") -> dict[str, Tensor]:  # type: ignore[name-defined]
        """Project each node type to *hidden_dim* vectors."""
        x_dict: dict[str, Tensor] = {}

        # Structural parts
        if "part" in data.node_types and data["part"].num_nodes > 0:
            cont = data["part"].x          # [N, 7]
            locs = cont[:, :2]             # [N, 2]
            other = cont[:, 2:]            # [N, 5]
            part_emb = self.part_id_embed(data["part"].part_id)   # [N, part_embed_dim]
            rot_emb = self.rotation_embed(data["part"].rotation)  # [N, 16]
            pe = self.loc_pe(locs)                                 # [N, pe_dim]
            feat = torch.cat([part_emb, rot_emb, pe, other], dim=1)
            x_dict["part"] = self.dropout(F.relu(self.part_proj(feat)))

        # Simple virtual types (x is [K, 1])
        for ntype in _SIMPLE_VIRT_TYPES:
            if ntype in data.node_types and data[ntype].num_nodes > 0:
                x_dict[ntype] = self.dropout(F.relu(self.simple_virt_projs[ntype](data[ntype].x)))

        # Zone types (x is [Z, 1], also has zone_label)
        for ntype, proj in (("zone", self.zone_proj), ("zone_rot", self.zone_rot_proj)):
            if ntype in data.node_types and data[ntype].num_nodes > 0:
                mc = data[ntype].x                                  # [Z, 1]
                zl = self.zone_label_embed(data[ntype].zone_label)  # [Z, 16]
                x_dict[ntype] = self.dropout(F.relu(proj(torch.cat([mc, zl], dim=1))))

        # Weapon groups (x is [W, 1], also has weapon_type)
        if "weapon_grp" in data.node_types and data["weapon_grp"].num_nodes > 0:
            mc = data["weapon_grp"].x                                         # [W, 1]
            wt = self.weapon_type_embed(data["weapon_grp"].weapon_type)       # [W, 16]
            x_dict["weapon_grp"] = self.dropout(
                F.relu(self.weapon_grp_proj(torch.cat([mc, wt], dim=1)))
            )

        return x_dict

    def _apply_virtual_dropout(self, x_dict: dict[str, Tensor]) -> dict[str, Tensor]:
        """Zero out each virtual node type's features with probability *virtual_dropout_rate*.

        Sets ``self._last_dropped_virtual_types`` so the training loop can skip
        edge-prediction losses for types whose features were zeroed.
        """
        if not self.training or self.virtual_dropout_rate <= 0.0:
            self._last_dropped_virtual_types = frozenset()
            return x_dict
        out = dict(x_dict)
        dropped: set[str] = set()
        device = next(iter(out.values())).device
        for ntype in _VIRTUAL_TYPES:
            if ntype in out and torch.rand(1, device=device).item() < self.virtual_dropout_rate:
                out[ntype] = torch.zeros_like(out[ntype])
                dropped.add(ntype)
        self._last_dropped_virtual_types = frozenset(dropped)
        return out

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(self, data: "HeteroData") -> dict[str, Tensor]:  # type: ignore[name-defined]
        """Run the full HGT encoder; returns node embeddings by type."""
        x_dict = self._encode(data)
        x_dict = self._apply_virtual_dropout(x_dict)

        # ship_info is a source-only node (sends messages, never receives them).
        # HGTConv drops it from its output, so we preserve the initial encoding
        # and re-inject it before each layer so it can keep sending messages.
        source_only = {"ship_info": x_dict["ship_info"]} if "ship_info" in x_dict else {}

        for conv in self.convs:
            # Filter to edge types whose src and tgt node types are present.
            # Must be recomputed each layer: HGTConv drops types that received
            # no messages, so x_dict keys may shrink between layers.
            edge_index_dict = {
                k: v
                for k, v in data.edge_index_dict.items()
                if k[0] in x_dict and k[2] in x_dict
            }
            x_dict = conv(x_dict, edge_index_dict)
            # HGTConv can return None for node types that received no messages.
            x_dict = {
                k: self.dropout(F.relu(v))
                for k, v in x_dict.items()
                if v is not None
            }
            # Restore source-only nodes with their fixed initial encoding.
            x_dict.update(source_only)

        return x_dict

    def predict_parts(self, x_dict: dict[str, Tensor]) -> Tensor:
        """Return part_id logits ``[N, num_classes]`` from encoded part features."""
        return self.part_pred_head(x_dict["part"])

    def predict_overclock(self, x_dict: dict[str, Tensor], mask_indices: Tensor) -> Tensor:
        """Binary overclock logits ``[M, 1]`` for masked part nodes."""
        return self.overclock_pred_head(x_dict["part"][mask_indices])

    def predict_edge(
        self,
        x_dict: dict[str, Tensor],
        src_type: str,
        src_indices: Tensor,
        tgt_type: str,
        tgt_indices: Tensor,
    ) -> Tensor:
        """Edge existence logits ``[E, 1]`` from concatenated endpoint embeddings."""
        src_emb = x_dict[src_type][src_indices]
        tgt_emb = x_dict[tgt_type][tgt_indices]
        return self.edge_pred_head(torch.cat([src_emb, tgt_emb], dim=-1))
