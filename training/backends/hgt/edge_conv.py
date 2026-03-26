"""EdgeAwareHGTConv: HGTConv fork with per-edge feature attention bias.

Forked from PyG 2.7.0 ``torch_geometric.nn.conv.hgt_conv.HGTConv``.
The only behavioral change is in ``message()``: an additive attention bias
computed from per-edge feature projections is applied alongside the existing
multiplicative ``p_rel`` scaling.  Edge types without features behave
identically to the original HGTConv.
"""

from __future__ import annotations

import math
from typing import Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
from torch import Tensor
from torch.nn import Parameter

from torch_geometric.nn.conv import MessagePassing
from torch_geometric.nn.dense import HeteroDictLinear, HeteroLinear
from torch_geometric.nn.inits import ones
from torch_geometric.nn.parameter_dict import ParameterDict
from torch_geometric.typing import Adj, EdgeType, Metadata, NodeType
from torch_geometric.utils import softmax
from torch_geometric.utils.hetero import construct_bipartite_edge_index

__all__ = ["EdgeAwareHGTConv"]


class EdgeAwareHGTConv(MessagePassing):
    """HGTConv with additive attention bias from per-edge features.

    For edge types listed in *edge_feat_groups*, a learned linear projection
    maps edge features to a per-head attention bias that is **added** to the
    dot-product attention logits before softmax.  All other behaviour
    (per-relation K/V transforms, ``p_rel`` multiplicative scaling, skip
    connections) is identical to upstream ``HGTConv``.

    Parameters
    ----------
    in_channels, out_channels, metadata, heads:
        Same as ``HGTConv``.
    edge_feat_groups:
        Mapping from a group name to ``(feat_dim, edge_type_keys)`` where
        *feat_dim* is the per-edge feature dimension and *edge_type_keys* is
        a sequence of ``'src__rel__dst'`` strings that share the same
        projector.  Example::

            {
                "touching": (1, ["part__touching__part"]),
                "travel":   (1, ["part__crew_access_reactor__part", ...]),
            }

        When ``None`` or empty, behaviour is identical to ``HGTConv``.
    """

    def __init__(
        self,
        in_channels: Union[int, Dict[str, int]],
        out_channels: int,
        metadata: Metadata,
        heads: int = 1,
        edge_feat_groups: dict[str, tuple[int, list[str]]] | None = None,
        **kwargs,
    ):
        super().__init__(aggr="add", node_dim=0, **kwargs)

        if out_channels % heads != 0:
            raise ValueError(
                f"'out_channels' (got {out_channels}) must be divisible by "
                f"the number of heads (got {heads})"
            )

        if not isinstance(in_channels, dict):
            in_channels = {node_type: in_channels for node_type in metadata[0]}

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.heads = heads
        self.node_types = metadata[0]
        self.edge_types = metadata[1]
        self.edge_types_map = {
            edge_type: i for i, edge_type in enumerate(metadata[1])
        }

        self.dst_node_types = {key[-1] for key in self.edge_types}

        self.kqv_lin = HeteroDictLinear(self.in_channels, self.out_channels * 3)
        self.out_lin = HeteroDictLinear(
            self.out_channels, self.out_channels, types=self.node_types
        )

        dim = out_channels // heads
        num_types = heads * len(self.edge_types)

        self.k_rel = HeteroLinear(dim, dim, num_types, bias=False, is_sorted=True)
        self.v_rel = HeteroLinear(dim, dim, num_types, bias=False, is_sorted=True)

        self.skip = ParameterDict(
            {node_type: Parameter(torch.empty(1)) for node_type in self.node_types}
        )

        self.p_rel = ParameterDict()
        for edge_type in self.edge_types:
            edge_type_str = "__".join(edge_type)
            self.p_rel[edge_type_str] = Parameter(torch.empty(1, heads))

        # --- Edge feature projectors ---
        # Reverse lookup: edge_type tuple → group name, for fast access in
        # _build_edge_feat_bias without per-forward string joining.
        self._edge_type_to_group: dict[tuple[str, str, str], str] = {}
        self.edge_feat_projs = nn.ModuleDict()
        if edge_feat_groups:
            for group_name, (feat_dim, edge_keys) in edge_feat_groups.items():
                self.edge_feat_projs[group_name] = nn.Linear(feat_dim, heads, bias=True)
                for ek in edge_keys:
                    parts = ek.split("__")
                    self._edge_type_to_group[(parts[0], parts[1], parts[2])] = group_name

        self.reset_parameters()

    def reset_parameters(self):
        super().reset_parameters()
        self.kqv_lin.reset_parameters()
        self.out_lin.reset_parameters()
        self.k_rel.reset_parameters()
        self.v_rel.reset_parameters()
        ones(self.skip)
        ones(self.p_rel)
        for proj in self.edge_feat_projs.values():
            # Initialize edge feature bias close to zero so the model starts
            # near-identical to vanilla HGTConv.
            nn.init.zeros_(proj.weight)
            nn.init.zeros_(proj.bias)

    # ------------------------------------------------------------------

    def _cat(self, x_dict: Dict[str, Tensor]) -> Tuple[Tensor, Dict[str, int]]:
        cumsum = 0
        outs: List[Tensor] = []
        offset: Dict[str, int] = {}
        for key, x in x_dict.items():
            outs.append(x)
            offset[key] = cumsum
            cumsum += x.size(0)
        return torch.cat(outs, dim=0), offset

    def _construct_src_node_feat(
        self,
        k_dict: Dict[str, Tensor],
        v_dict: Dict[str, Tensor],
        edge_index_dict: Dict[EdgeType, Adj],
    ) -> Tuple[Tensor, Tensor, Dict[EdgeType, int]]:
        cumsum = 0
        num_edge_types = len(self.edge_types)
        H, D = self.heads, self.out_channels // self.heads

        ks: List[Tensor] = []
        vs: List[Tensor] = []
        type_list: List[Tensor] = []
        offset: Dict[EdgeType, int] = {}
        for edge_type in edge_index_dict.keys():
            src = edge_type[0]
            N = k_dict[src].size(0)
            offset[edge_type] = cumsum
            cumsum += N

            edge_type_offset = self.edge_types_map[edge_type]
            type_vec = (
                torch.arange(H, dtype=torch.long).view(-1, 1).repeat(1, N)
                * num_edge_types
                + edge_type_offset
            )

            type_list.append(type_vec)
            ks.append(k_dict[src])
            vs.append(v_dict[src])

        ks = torch.cat(ks, dim=0).transpose(0, 1).reshape(-1, D)
        vs = torch.cat(vs, dim=0).transpose(0, 1).reshape(-1, D)
        type_vec = torch.cat(type_list, dim=1).flatten()

        k = self.k_rel(ks, type_vec).view(H, -1, D).transpose(0, 1)
        v = self.v_rel(vs, type_vec).view(H, -1, D).transpose(0, 1)

        return k, v, offset

    # ------------------------------------------------------------------

    def _build_edge_feat_bias(
        self,
        src_offset: Dict[EdgeType, int],
        edge_index_dict: Dict[EdgeType, Adj],
        edge_attr_dict: Dict[EdgeType, Tensor] | None,
        total_edges: int,
        device: torch.device,
    ) -> Tensor | None:
        """Build a ``[total_edges, H]`` additive attention bias from edge features.

        Returns ``None`` when no edge features are available so the caller can
        skip the addition entirely.
        """
        if not self._edge_type_to_group or not edge_attr_dict:
            return None

        bias: Tensor | None = None
        offset = 0
        for edge_type in src_offset:
            n_edges = edge_index_dict[edge_type].shape[1]
            group = self._edge_type_to_group.get(edge_type)
            if group is not None and edge_type in edge_attr_dict:
                if bias is None:
                    bias = torch.zeros(total_edges, self.heads, device=device)
                feat = edge_attr_dict[edge_type]  # [n_edges, feat_dim]
                bias[offset : offset + n_edges] = self.edge_feat_projs[group](feat)
            offset += n_edges

        return bias

    # ------------------------------------------------------------------

    def forward(
        self,
        x_dict: Dict[NodeType, Tensor],
        edge_index_dict: Dict[EdgeType, Adj],
        edge_attr_dict: Dict[EdgeType, Tensor] | None = None,
    ) -> Dict[NodeType, Optional[Tensor]]:
        """Run forward pass with optional per-edge feature attention bias.

        Parameters
        ----------
        x_dict:
            Node features by type.
        edge_index_dict:
            Edge connectivity by type.
        edge_attr_dict:
            Per-edge feature tensors by type.  Only edge types registered in
            ``edge_feat_groups`` at construction time are used; others are
            silently ignored.  Pass ``None`` to skip edge features entirely.
        """
        H = self.heads
        D = self.out_channels // H

        k_dict: Dict[str, Tensor] = {}
        q_dict: Dict[str, Tensor] = {}
        v_dict: Dict[str, Tensor] = {}
        out_dict: Dict[str, Tensor] = {}

        kqv_dict = self.kqv_lin(x_dict)
        for key, val in kqv_dict.items():
            k, q, v = torch.tensor_split(val, 3, dim=1)
            k_dict[key] = k.view(-1, H, D)
            q_dict[key] = q.view(-1, H, D)
            v_dict[key] = v.view(-1, H, D)

        q, dst_offset = self._cat(q_dict)
        k, v, src_offset = self._construct_src_node_feat(
            k_dict, v_dict, edge_index_dict
        )

        edge_index, edge_attr = construct_bipartite_edge_index(
            edge_index_dict,
            src_offset,
            dst_offset,
            edge_attr_dict=self.p_rel,
            num_nodes=k.size(0),
        )

        # Build additive attention bias from per-edge features.
        total_edges = edge_index.shape[1]
        device = k.device
        edge_feat_bias = self._build_edge_feat_bias(
            src_offset, edge_index_dict, edge_attr_dict, total_edges, device
        )

        out = self.propagate(
            edge_index,
            k=k,
            q=q,
            v=v,
            edge_attr=edge_attr,
            edge_feat_bias=edge_feat_bias,
        )

        # Reconstruct output node embeddings dict.
        for node_type, start_offset in dst_offset.items():
            end_offset = start_offset + q_dict[node_type].size(0)
            if node_type in self.dst_node_types:
                out_dict[node_type] = out[start_offset:end_offset]

        # Transform output node embeddings.
        a_dict = self.out_lin(
            {
                k: torch.nn.functional.gelu(v) if v is not None else v
                for k, v in out_dict.items()
            }
        )

        # Skip connection with gating.
        for node_type, out in out_dict.items():
            out = a_dict[node_type]
            if out.size(-1) == x_dict[node_type].size(-1):
                alpha = self.skip[node_type].sigmoid()
                out = alpha * out + (1 - alpha) * x_dict[node_type]
            out_dict[node_type] = out

        return out_dict

    def message(
        self,
        k_j: Tensor,
        q_i: Tensor,
        v_j: Tensor,
        edge_attr: Tensor,
        edge_feat_bias: Tensor | None,
        index: Tensor,
        ptr: Optional[Tensor],
        size_i: Optional[int],
    ) -> Tensor:
        alpha = (q_i * k_j).sum(dim=-1) * edge_attr
        if edge_feat_bias is not None:
            alpha = alpha + edge_feat_bias
        alpha = alpha / math.sqrt(q_i.size(-1))
        alpha = softmax(alpha, index, ptr, size_i)
        out = v_j * alpha.view(-1, self.heads, 1)
        return out.view(-1, self.out_channels)

    def __repr__(self) -> str:
        groups = list(self.edge_feat_projs.keys()) if self.edge_feat_projs else []
        return (
            f"{self.__class__.__name__}(-1, {self.out_channels}, "
            f"heads={self.heads}, edge_feat_groups={groups})"
        )
