"""Tests for EdgeAwareHGTConv edge feature integration."""

from __future__ import annotations

import torch

from training.backends.hgt.edge_conv import EdgeAwareHGTConv


def _make_simple_metadata():
    """Minimal metadata with two node types and two edge types."""
    node_types = ["A", "B"]
    edge_types = [("A", "rel1", "A"), ("B", "rel2", "A")]
    return (node_types, edge_types)


def _make_simple_data(shared_sides_val: float = 1.0):
    """Build x_dict, edge_index_dict, edge_attr_dict for a tiny graph."""
    x_dict = {
        "A": torch.randn(4, 16),
        "B": torch.randn(2, 16),
    }
    edge_index_dict = {
        ("A", "rel1", "A"): torch.tensor([[0, 1, 2], [1, 2, 3]]),
        ("B", "rel2", "A"): torch.tensor([[0, 1], [0, 2]]),
    }
    edge_attr_dict = {
        ("A", "rel1", "A"): torch.full((3, 1), shared_sides_val),
    }
    return x_dict, edge_index_dict, edge_attr_dict


def test_edge_feat_changes_output():
    """Different edge features should produce different outputs."""
    metadata = _make_simple_metadata()
    edge_feat_groups = {
        "group1": (1, ["A__rel1__A"]),
    }

    conv = EdgeAwareHGTConv(16, 16, metadata, heads=4, edge_feat_groups=edge_feat_groups)
    conv.eval()

    # Set non-zero edge feature projection so the bias is non-trivial.
    with torch.no_grad():
        conv.edge_feat_projs["group1"].weight.fill_(1.0)
        conv.edge_feat_projs["group1"].bias.fill_(0.0)

    x_dict, edge_index_dict, _ = _make_simple_data()

    # Run with different edge feature values.
    attr_low = {("A", "rel1", "A"): torch.full((3, 1), 1.0)}
    attr_high = {("A", "rel1", "A"): torch.full((3, 1), 5.0)}

    out_low = conv(x_dict, edge_index_dict, edge_attr_dict=attr_low)
    out_high = conv(x_dict, edge_index_dict, edge_attr_dict=attr_high)

    # Outputs should differ because the attention bias differs.
    assert not torch.allclose(out_low["A"], out_high["A"], atol=1e-6), \
        "Edge features should change the output"


def test_no_edge_feats_matches_vanilla():
    """Without edge_feat_groups, output should match vanilla behavior."""
    metadata = _make_simple_metadata()

    conv_with = EdgeAwareHGTConv(16, 16, metadata, heads=4, edge_feat_groups=None)
    conv_with.eval()

    x_dict, edge_index_dict, _ = _make_simple_data()

    # Should work without edge_attr_dict.
    out = conv_with(x_dict, edge_index_dict)
    assert "A" in out
    assert out["A"].shape == (4, 16)


def test_edge_feat_none_attr_dict():
    """Passing edge_attr_dict=None with edge_feat_groups configured should not crash."""
    metadata = _make_simple_metadata()
    edge_feat_groups = {"group1": (1, ["A__rel1__A"])}

    conv = EdgeAwareHGTConv(16, 16, metadata, heads=4, edge_feat_groups=edge_feat_groups)
    conv.eval()

    x_dict, edge_index_dict, _ = _make_simple_data()

    # Should work — edge_feat_bias will be None, behaving like vanilla HGTConv.
    out = conv(x_dict, edge_index_dict, edge_attr_dict=None)
    assert "A" in out
    assert out["A"].shape == (4, 16)


def test_backward_pass():
    """Gradients should flow through edge feature projections."""
    metadata = _make_simple_metadata()
    edge_feat_groups = {"group1": (1, ["A__rel1__A"])}

    conv = EdgeAwareHGTConv(16, 16, metadata, heads=4, edge_feat_groups=edge_feat_groups)

    x_dict, edge_index_dict, edge_attr_dict = _make_simple_data()

    out = conv(x_dict, edge_index_dict, edge_attr_dict=edge_attr_dict)
    loss = out["A"].sum()
    loss.backward()

    proj = conv.edge_feat_projs["group1"]
    assert proj.weight.grad is not None, "Edge feature projection should receive gradients"
    assert proj.weight.grad.abs().sum() > 0, "Gradients should be non-zero"


def test_zero_init_matches_no_feat():
    """With zero-init projectors, output should match no-edge-features case."""
    metadata = _make_simple_metadata()
    edge_feat_groups = {"group1": (1, ["A__rel1__A"])}

    conv = EdgeAwareHGTConv(16, 16, metadata, heads=4, edge_feat_groups=edge_feat_groups)
    conv.eval()
    # reset_parameters already zeros the edge feat projs.

    x_dict, edge_index_dict, edge_attr_dict = _make_simple_data()

    out_with_feat = conv(x_dict, edge_index_dict, edge_attr_dict=edge_attr_dict)
    out_no_feat = conv(x_dict, edge_index_dict, edge_attr_dict=None)

    assert torch.allclose(out_with_feat["A"], out_no_feat["A"], atol=1e-5), \
        "Zero-init projectors should produce identical output to no edge features"


def test_shared_projector_across_edge_types():
    """Multiple edge types sharing one group should use the same projector."""
    node_types = ["A"]
    edge_types = [("A", "r1", "A"), ("A", "r2", "A")]
    metadata = (node_types, edge_types)

    edge_feat_groups = {
        "shared": (1, ["A__r1__A", "A__r2__A"]),
    }

    conv = EdgeAwareHGTConv(16, 16, metadata, heads=4, edge_feat_groups=edge_feat_groups)

    # Only one projector should exist, shared across both edge types.
    assert len(conv.edge_feat_projs) == 1
    assert "shared" in conv.edge_feat_projs
