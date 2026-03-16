"""Weapon grouping expansion pass.

This module defines a structural expansion pass that identifies weapon parts,
groups them by weapon type, and emits weapon-group virtual nodes with
membership cross-edges into the structural expansion graph.
"""

from __future__ import annotations

from typing import Any, Dict, List, Mapping, MutableMapping

from graph_expansion.context import ExpansionContext
from graph_expansion.passes.base import ExpansionPass

__all__ = ["WeaponGroupsPass", "WEAPON_TYPE_SUBSTRINGS"]

_EXPANSION_GRAPH_NAME = "X_expansion_structural"

# Ordered list of substrings used to detect weapon types from part IDs.
WEAPON_TYPE_SUBSTRINGS: List[str] = [
    "cannon",
    "railgun",
    "missile_launcher",
    "laser_blaster",
    "chaingun",
    "disruptor",
    "resonance_beam",
    "ion_beam_emitter",
    "point_defense",
    "tractor_beam",
    "mining_laser",
    "manipulator_beam",
]


def _detect_weapon_type(part_id: str) -> str | None:
    """Return the weapon type substring for *part_id*, or None when not a weapon.

    Args:
        part_id: Identifier of the structural part node.

    Returns:
        The first matching weapon-type substring from ``WEAPON_TYPE_SUBSTRINGS``
        when *part_id* is identified as a weapon, otherwise ``None``.
    """

    lower_id = part_id.lower()
    for token in WEAPON_TYPE_SUBSTRINGS:
        # Check for the first substring match in priority order.
        if token in lower_id:
            return token
    return None


class WeaponGroupsPass(ExpansionPass):
    """Emit weapon-group virtual nodes and membership cross-edges."""

    name = "weapon_groups"
    version = 1
    requires = ("base_indexes",)
    provides = ("weapon_group_by_part_id",)

    def run(self, context: ExpansionContext) -> Mapping[str, Any]:
        """Group weapon parts and emit weapon-group nodes and edges.

        This pass inspects structural nodes to find weapon parts, groups them
        by weapon type, records a per-part weapon type annotation, and then
        materializes weapon-group nodes with ``weapon_member`` cross-edges
        into the structural expansion graph.
        """

        # Prefer cached structural nodes produced by earlier passes.
        structural_nodes = context.caches.get("structural_nodes")
        if structural_nodes is None:
            structural_graph = context.get_source_graph("A_structural_part_graph")
            structural_nodes = list(structural_graph.get("nodes", []))

        # Build mapping from weapon type to sorted list of member node IDs.
        weapon_type_to_member_ids: Dict[str, List[int]] = {}
        weapon_group_by_part_id: Dict[int, str] = {}

        for node in structural_nodes:
            part_id = node.get("part_id")
            if not isinstance(part_id, str):
                continue

            weapon_type = _detect_weapon_type(part_id)
            if weapon_type is None:
                continue

            node_id = int(node["id"])
            weapon_group_by_part_id[node_id] = weapon_type
            weapon_type_to_member_ids.setdefault(weapon_type, []).append(node_id)

        # Ensure members are emitted in deterministic ID order for each group.
        for member_ids in weapon_type_to_member_ids.values():
            member_ids.sort()

        # Persist the per-part weapon type annotation on the context.
        context.set_annotation("weapon_group_by_part_id", weapon_group_by_part_id)

        # Acquire the expansion graph and hooks for nodes and cross-edges.
        expansion_graph = context.ensure_emitted_graph(_EXPANSION_GRAPH_NAME)
        nodes: List[MutableMapping[str, Any]] = expansion_graph["nodes"]
        cross_edges: List[MutableMapping[str, Any]] = expansion_graph["cross_edges"]

        weapon_group_nodes: List[Dict[str, Any]] = []
        weapon_member_edges: List[Dict[str, Any]] = []

        # Emit one weapon-group node per weapon type that has at least one member.
        for weapon_type in WEAPON_TYPE_SUBSTRINGS:
            member_ids = weapon_type_to_member_ids.get(weapon_type)
            if not member_ids:
                continue

            group_id = f"weapon_group_{weapon_type}"
            weapon_group_nodes.append(
                {
                    "id": group_id,
                    "kind": "weapon_group",
                    "weapon_type": weapon_type,
                    "member_count": len(member_ids),
                }
            )

            # Emit membership cross-edges from the group node to each weapon node.
            for member_id in member_ids:
                weapon_member_edges.append(
                    {
                        "source": group_id,
                        "source_graph": _EXPANSION_GRAPH_NAME,
                        "target": member_id,
                        "target_graph": "A_structural_part_graph",
                        "kind": "weapon_member",
                    }
                )

        # Append newly created nodes and edges to the expansion graph.
        nodes.extend(weapon_group_nodes)
        cross_edges.extend(weapon_member_edges)

        # Update expansion-graph summary counters with weapon-specific fields.
        summary = expansion_graph.setdefault("summary", {})
        summary.setdefault("weapon_group_nodes", 0)
        summary.setdefault("weapon_member_edges", 0)
        summary["weapon_group_nodes"] += len(weapon_group_nodes)
        summary["weapon_member_edges"] += len(weapon_member_edges)

        return {
            "weapon_group_nodes": len(weapon_group_nodes),
            "weapon_member_edges": len(weapon_member_edges),
        }

