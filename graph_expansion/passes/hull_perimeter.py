"""Hull perimeter classification expansion pass.

This module defines an expansion pass that classifies structural part nodes
in the ``A_structural_part_graph`` as either hull-perimeter parts or interior
parts based on their occupied cells in the centered-2x coordinate frame.
"""

from __future__ import annotations

from typing import Any, Dict, List, Mapping, MutableMapping, Sequence, Set, Tuple

from graph_expansion.context import EXPANSION_GRAPH_NAME, STRUCTURAL_GRAPH_NAME, ExpansionContext
from graph_expansion.passes.base import ExpansionPass

__all__ = ["HullPerimeterPass"]


def _compute_footprint_cells_2x(node: Mapping[str, Any]) -> Set[Tuple[int, int]]:
    """Compute the set of 2x-grid footprint cells occupied by *node*.

    The computation uses the node ``location_2x``, ``rotation``, and
    ``footprint`` fields as described in the hull-perimeter specification.
    """

    location_2x = node.get("location_2x")
    footprint = node.get("footprint")

    # Guard against missing data by returning an empty set when required
    # attributes are unavailable or malformed
    if not isinstance(location_2x, Sequence) or len(location_2x) != 2:
        return set()
    if not isinstance(footprint, Mapping):
        return set()

    lx, ly = int(location_2x[0]), int(location_2x[1])
    rotation = int(node.get("rotation", 0)) % 4

    width = int(footprint.get("width", 0))
    height = int(footprint.get("height", 0))
    if width <= 0 or height <= 0:
        return set()

    # Determine effective dimensions after rotation; 90-degree rotations swap
    # width and height in the 2x grid
    if rotation % 2 == 0:
        effective_width = width
        effective_height = height
    else:
        effective_width = height
        effective_height = width

    cells: Set[Tuple[int, int]] = set()
    # Enumerate all footprint cells in the centered-2x frame
    for row in range(effective_height):
        for col in range(effective_width):
            cell_x = lx + 2 * col
            cell_y = ly + 2 * row
            cells.add((cell_x, cell_y))

    return cells


class HullPerimeterPass(ExpansionPass):
    """Classify structural parts as hull perimeter or interior."""

    name = "hull_perimeter"
    version = 1
    requires = ("base_indexes",)
    provides = ("hull_role_by_part_id",)

    def run(self, context: ExpansionContext) -> Mapping[str, Any]:
        """Run the hull-perimeter classification and emit virtual nodes.

        Args:
            context: Expansion context for the current source artifact.

        Returns:
            Mapping with counts for hull-perimeter and interior parts.
        """

        # Retrieve structural nodes from the shared cache populated by
        # BaseIndexesPass, falling back to loading the source graph when
        # necessary for robustness
        structural_nodes = context.caches.get("structural_nodes")
        if structural_nodes is None:
            structural_graph = context.get_source_graph(STRUCTURAL_GRAPH_NAME)
            structural_nodes = list(structural_graph.get("nodes", []))

        # Build a global set of all occupied cells across every structural part
        occupied_cells: Set[Tuple[int, int]] = set()
        node_cells_by_id: Dict[int, Set[Tuple[int, int]]] = {}
        for node in structural_nodes:
            node_id = node.get("id")
            if not isinstance(node_id, int):
                continue
            node_cells = _compute_footprint_cells_2x(node)
            node_cells_by_id[node_id] = node_cells
            occupied_cells.update(node_cells)

        # Determine whether each part lies on the hull perimeter or in the
        # interior by checking for at least one unoccupied neighbor cell
        hull_role_by_part_id: Dict[int, str] = {}
        perimeter_part_ids: List[int] = []
        interior_part_ids: List[int] = []

        neighbor_offsets: Tuple[Tuple[int, int], ...] = ((2, 0), (-2, 0), (0, 2), (0, -2))

        for node in structural_nodes:
            node_id = node.get("id")
            if not isinstance(node_id, int):
                continue
            node_cells = node_cells_by_id.get(node_id, set())

            # Default nodes without any footprint cells to interior so that the
            # annotation covers all parts
            if not node_cells:
                hull_role_by_part_id[node_id] = "interior"
                interior_part_ids.append(node_id)
                continue

            is_perimeter = False
            for cell_x, cell_y in node_cells:
                for dx, dy in neighbor_offsets:
                    neighbor = (cell_x + dx, cell_y + dy)
                    if neighbor not in occupied_cells:
                        is_perimeter = True
                        break
                if is_perimeter:
                    break

            if is_perimeter:
                hull_role_by_part_id[node_id] = "perimeter"
                perimeter_part_ids.append(node_id)
            else:
                hull_role_by_part_id[node_id] = "interior"
                interior_part_ids.append(node_id)

        # Store the per-part hull role annotation for use by later passes
        context.set_annotation("hull_role_by_part_id", hull_role_by_part_id)

        # Materialize virtual hull-perimeter and interior nodes plus their
        # cross-edges in the structural expansion graph
        expansion_graph = context.ensure_emitted_graph(EXPANSION_GRAPH_NAME)
        nodes: List[MutableMapping[str, Any]] = expansion_graph["nodes"]
        cross_edges: List[MutableMapping[str, Any]] = expansion_graph["cross_edges"]

        hull_perimeter_node_id = "hull_perimeter"
        interior_node_id = "interior"

        hull_perimeter_node: Dict[str, Any] = {
            "id": hull_perimeter_node_id,
            "kind": "hull_perimeter",
            "member_count": len(perimeter_part_ids),
        }
        interior_node: Dict[str, Any] = {
            "id": interior_node_id,
            "kind": "hull_interior",
            "member_count": len(interior_part_ids),
        }

        hull_perimeter_edges: List[Dict[str, Any]] = []
        for part_id in perimeter_part_ids:
            hull_perimeter_edges.append(
                {
                    "source": hull_perimeter_node_id,
                    "source_graph": EXPANSION_GRAPH_NAME,
                    "target": part_id,
                    "target_graph": STRUCTURAL_GRAPH_NAME,
                    "kind": "hull_member",
                }
            )

        interior_edges: List[Dict[str, Any]] = []
        for part_id in interior_part_ids:
            interior_edges.append(
                {
                    "source": interior_node_id,
                    "source_graph": EXPANSION_GRAPH_NAME,
                    "target": part_id,
                    "target_graph": STRUCTURAL_GRAPH_NAME,
                    "kind": "interior_member",
                }
            )

        nodes.extend([hull_perimeter_node, interior_node])
        cross_edges.extend(hull_perimeter_edges)
        cross_edges.extend(interior_edges)

        context.increment_summary(
            EXPANSION_GRAPH_NAME,
            hull_perimeter_parts=len(perimeter_part_ids),
            interior_parts=len(interior_part_ids),
        )

        return {
            "hull_perimeter_parts": len(perimeter_part_ids),
            "interior_parts": len(interior_part_ids),
        }

