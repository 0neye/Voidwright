"""ShipHGT: Heterogeneous Graph Transformer encoder for ship designs."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch_geometric.nn import HGTConv

from training.backends.hgt.convert import build_metadata
from training.backends.hgt.vocab import WEAPON_TYPES

__all__ = ["ShipHGT", "SinusoidalPE"]

ROTATION_MASK_IDX = 4

# Virtual node types that can be dropped during training as conditioning dropout.
_VIRTUAL_TYPES: tuple[str, ...] = (
    "cluster", "thermal",
    "zone", "zone_rot", "weapon_grp", "ship_info",
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
       heterogeneous message passing with per-relation K/V transforms.
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
        num_layers: int = 2,
        dropout: float = 0.1,
        pe_dim: int = 32,
        virtual_dropout_rate: float = 0.3,
        reverse_edges: bool = True,
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
        self.rotation_embed = nn.Embedding(5, 16)
        self.loc_pe = SinusoidalPE(pe_dim)
        # part input: part_id_embed + rotation_embed + pe + [fp_cells, fp_w, fp_h, traversable, overclocked]
        part_in_dim = part_embed_dim + 16 + pe_dim + 5
        self.part_proj = nn.Linear(part_in_dim, hidden_dim)

        # --- Input projections: virtual nodes ---

        # ship_info: 8 features → hidden_dim
        # [log1p(total_parts), log1p(occupied_cells), footprint_w_2x, footprint_h_2x,
        #  log1p(cluster_count), log1p(thermal_count), log1p(weapon_grp_count), log1p(zone_count)]
        self.ship_info_proj = nn.Linear(8, hidden_dim)

        # cluster: 5 features → hidden_dim
        # [log1p(member_count), log1p(door_count), log1p(walkable_cells_2x), centroid_x, centroid_y]
        self.cluster_proj = nn.Linear(5, hidden_dim)

        # thermal: 4 features → hidden_dim
        # [log1p(member_count), log1p(backbone_count), log1p(overclocked_count), leaf_fraction]
        self.thermal_proj = nn.Linear(4, hidden_dim)

        # zone / zone_rot: 3 features + zone_label_embed(16) → hidden_dim
        # [log1p(member_count), log1p(occupied_cells), avg_radius_2x]
        self.zone_label_embed = nn.Embedding(8, 16)
        self.zone_proj = nn.Linear(3 + 16, hidden_dim)
        self.zone_rot_proj = nn.Linear(3 + 16, hidden_dim)

        # weapon_grp: 4 features + weapon_type_embed(16) → hidden_dim
        # [log1p(member_count), centroid_x, centroid_y, log1p(spatial_spread)]
        self.weapon_type_embed = nn.Embedding(len(WEAPON_TYPES), 16)
        self.weapon_grp_proj = nn.Linear(4 + 16, hidden_dim)

        # --- HGTConv layers ---
        # HGTConv uses per-relation K/Q/V linear transforms (HeteroLinear) for
        # each (head, edge_type) pair — significantly more expressive than
        # grouped parameter sharing.  build_metadata() ensures only edge types
        # that will actually appear in the data get allocated transforms.
        metadata = build_metadata(reverse_edges=reverse_edges)
        self.convs = nn.ModuleList(
            [HGTConv(hidden_dim, hidden_dim, metadata=metadata, heads=num_heads)
             for _ in range(num_layers)]
        )

        self.dropout = nn.Dropout(dropout)

        # --- Masked part prediction head ---
        # Predicts over num_classes (all valid part_ids + <unk>, excludes <mask>).
        self.part_pred_head = nn.Linear(hidden_dim, vocab_size - 1)

        # --- Auxiliary masked prediction heads ---
        # 4-class rotation prediction for masked parts.
        self.rotation_pred_head = nn.Linear(hidden_dim, 4)
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

        # ship_info: 8-feature conditioning node
        if "ship_info" in data.node_types and data["ship_info"].num_nodes > 0:
            x_dict["ship_info"] = self.dropout(F.relu(self.ship_info_proj(data["ship_info"].x)))

        # cluster: 5-feature traversable cluster nodes
        if "cluster" in data.node_types and data["cluster"].num_nodes > 0:
            x_dict["cluster"] = self.dropout(F.relu(self.cluster_proj(data["cluster"].x)))

        # thermal: 4-feature thermal network nodes
        if "thermal" in data.node_types and data["thermal"].num_nodes > 0:
            x_dict["thermal"] = self.dropout(F.relu(self.thermal_proj(data["thermal"].x)))

        # zone / zone_rot: 3-feature + zone label embedding
        for ntype, proj in (("zone", self.zone_proj), ("zone_rot", self.zone_rot_proj)):
            if ntype in data.node_types and data[ntype].num_nodes > 0:
                feat = data[ntype].x                                    # [Z, 3]
                zl = self.zone_label_embed(data[ntype].zone_label)      # [Z, 16]
                x_dict[ntype] = self.dropout(F.relu(proj(torch.cat([feat, zl], dim=1))))

        # weapon_grp: 4-feature + weapon type embedding
        if "weapon_grp" in data.node_types and data["weapon_grp"].num_nodes > 0:
            feat = data["weapon_grp"].x                                          # [W, 4]
            wt = self.weapon_type_embed(data["weapon_grp"].weapon_type)          # [W, 16]
            x_dict["weapon_grp"] = self.dropout(
                F.relu(self.weapon_grp_proj(torch.cat([feat, wt], dim=1)))
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
        # HGTConv omits it from its output when it has no incoming edges, so
        # preserve the initial encoding and re-inject it before each layer.
        source_only = {"ship_info": x_dict["ship_info"]} if "ship_info" in x_dict else {}

        dropped = self._last_dropped_virtual_types

        for conv in self.convs:
            # Filter to edge types whose src and tgt are present AND not dropped.
            # Excluding dropped types' edges prevents noisy gradient flow through
            # disconnected virtual nodes with zero embeddings.
            try:
                raw_edges = data.edge_index_dict.items()
            except KeyError:
                raw_edges = []  # batch has no edges in any edge type
            edge_index_dict = {
                k: v
                for k, v in raw_edges
                if k[0] in x_dict and k[2] in x_dict
                and k[0] not in dropped and k[2] not in dropped
                and v.shape[1] > 0
            }
            # HGTConv crashes on an empty edge dict (all types dropped by virtual
            # dropout or absent in this batch). Skip message passing and keep the
            # current embeddings; the prediction head still runs on them.
            if edge_index_dict:
                x_dict = conv(x_dict, edge_index_dict)
                x_dict = {k: self.dropout(F.relu(v)) for k, v in x_dict.items()}
            # Restore source-only nodes with their fixed initial encoding.
            x_dict.update(source_only)

        return x_dict

    def predict_parts(self, x_dict: dict[str, Tensor]) -> Tensor:
        """Return part_id logits ``[N, num_classes]`` from encoded part features."""
        return self.part_pred_head(x_dict["part"])

    def predict_rotation(self, x_dict: dict[str, Tensor], mask_indices: Tensor) -> Tensor:
        """4-class rotation logits ``[M, 4]`` for masked part nodes."""
        return self.rotation_pred_head(x_dict["part"][mask_indices])

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
