"""Training and evaluation loop for the HGT encoder."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING

import torch
import torch.nn.functional as F
from torch import Tensor

from training.backends.hgt.convert import DOOR_EDGE_KEY
from training.backends.hgt.model import ROTATION_MASK_IDX

if TYPE_CHECKING:
    from torch_geometric.data import HeteroData

    from training.backends.hgt.model import ShipHGT

__all__ = [
    "apply_part_mask",
    "apply_rotation_mask",
    "apply_overclock_mask",
    "apply_edge_mask",
    "train_epoch",
    "eval_epoch",
    "load_dataset",
    "save_checkpoint",
    "load_checkpoint",
]

log = logging.getLogger(__name__)

OVERCLOCK_MASK_VALUE = -1.0

# Edge types eligible for virtual-edge masked link prediction.
_VIRTUAL_EDGE_TYPES: tuple[tuple[str, str, str], ...] = (
    ("cluster", "super_member", "part"),
    ("thermal", "thermal_member", "part"),
    ("zone", "zone_member", "part"),
    ("zone_rot", "zone_member_rotated", "part"),
    ("weapon_grp", "weapon_member", "part"),
)


# ---------------------------------------------------------------------------
# Masking helpers
# ---------------------------------------------------------------------------

def apply_part_mask(
    data: "HeteroData",
    mask_rate: float,
    mask_token_idx: int,
) -> tuple["HeteroData", Tensor, Tensor]:
    """Randomly replace *mask_rate* fraction of part_id values with *mask_token_idx*.

    Returns
    -------
    data:
        Modified (in-place) batch with masked part_id values.
    mask_indices:
        1-D LongTensor of the positions that were masked.
    true_labels:
        1-D LongTensor of the original part_id values at those positions.
    """
    n = data["part"].num_nodes
    num_mask = max(1, int(n * mask_rate))
    mask_indices = torch.randperm(n, device=data["part"].part_id.device)[:num_mask]
    true_labels = data["part"].part_id[mask_indices].clone()
    # Clone before mutation to avoid corrupting the cached dataset.
    data["part"].part_id = data["part"].part_id.clone()
    data["part"].part_id[mask_indices] = mask_token_idx
    return data, mask_indices, true_labels


def apply_rotation_mask(
    data: "HeteroData",
    mask_rate: float,
) -> tuple["HeteroData", Tensor, Tensor]:
    """Independently mask rotation on a random subset of part nodes.

    Unlike part_id masking, the rotation mask is applied to a *separate* set of
    nodes chosen independently.  This means most rotation-masked nodes still
    have their part_id visible, giving the model richer context for learning
    rotation prediction while avoiding the heavy input corruption of joint
    masking.

    Returns
    -------
    data:
        Modified batch with masked rotation sentinel values.
    mask_indices:
        1-D LongTensor of the positions that were masked.
    true_labels:
        1-D LongTensor of original rotation values (0..3) at those positions.
    """
    n = data["part"].num_nodes
    num_mask = max(1, int(n * mask_rate))
    mask_indices = torch.randperm(n, device=data["part"].rotation.device)[:num_mask]
    true_labels = data["part"].rotation[mask_indices].clone()
    data["part"].rotation = data["part"].rotation.clone()
    data["part"].rotation[mask_indices] = ROTATION_MASK_IDX
    return data, mask_indices, true_labels


def apply_overclock_mask(
    data: "HeteroData",
    mask_rate: float,
) -> tuple["HeteroData", Tensor, Tensor]:
    """Replace the overclocked flag with a dedicated sentinel on masked parts.

    Returns
    -------
    data:
        Modified batch with overclocked features masked for masked nodes.
    mask_indices:
        1-D LongTensor of the part positions that were masked.
    true_labels:
        1-D FloatTensor of original overclocked values (0.0 or 1.0).
    """
    n = data["part"].num_nodes
    num_mask = max(1, int(n * mask_rate))
    device = data["part"].x.device
    mask_indices = torch.randperm(n, device=device)[:num_mask]
    true_labels = data["part"].x[mask_indices, 6].clone()
    # Clone before mutation to avoid corrupting the cached dataset.
    data["part"].x = data["part"].x.clone()
    data["part"].x[mask_indices, 6] = OVERCLOCK_MASK_VALUE
    return data, mask_indices, true_labels


def apply_edge_mask(
    data: "HeteroData",
    edge_key: tuple[str, str, str],
    mask_rate: float,
) -> tuple["HeteroData", Tensor, Tensor]:
    """Remove a random fraction of edges of *edge_key* before message passing.

    Edges are removed so the model cannot observe them during the forward pass,
    making edge prediction a genuine reconstruction task.

    Returns
    -------
    data:
        Modified batch with masked edges removed.
    masked_src:
        1-D LongTensor of global source indices of the removed edges.
    masked_tgt:
        1-D LongTensor of global target indices of the removed edges.
    """
    store = data[edge_key]
    ei: Tensor | None = getattr(store, "edge_index", None)
    device = data["part"].part_id.device
    empty = torch.zeros(0, dtype=torch.long, device=device)
    if ei is None or ei.shape[1] == 0:
        return data, empty, empty

    E = ei.shape[1]
    num_mask = max(1, int(E * mask_rate))
    perm = torch.randperm(E, device=ei.device)
    mask_idx = perm[:num_mask]
    keep_idx = perm[num_mask:]

    masked_src = ei[0, mask_idx].clone()
    masked_tgt = ei[1, mask_idx].clone()
    data[edge_key].edge_index = ei[:, keep_idx].clone()
    edge_attr: Tensor | None = getattr(store, "edge_attr", None)
    if edge_attr is not None and edge_attr.shape[0] > 0:
        data[edge_key].edge_attr = edge_attr[keep_idx].clone()

    return data, masked_src, masked_tgt


def _sample_negative_edges(
    n_src: int,
    n_tgt: int,
    pos_encoded: Tensor,
    n_neg: int,
    device: torch.device,
) -> tuple[Tensor, Tensor]:
    """Sample *n_neg* random (src, tgt) index pairs absent from the positive edge set.

    Parameters
    ----------
    pos_encoded:
        1-D LongTensor of positive pairs encoded as ``src * n_tgt + tgt``.
    """
    if n_neg == 0 or n_src == 0 or n_tgt == 0:
        empty = torch.zeros(0, dtype=torch.long, device=device)
        return empty, empty

    collected_s: list[Tensor] = []
    collected_t: list[Tensor] = []
    n_collected = 0

    for _ in range(20):  # at most 20 sampling rounds
        batch_n = min(n_neg * 4, 8192)
        s = torch.randint(0, n_src, (batch_n,), device=device)
        t = torch.randint(0, n_tgt, (batch_n,), device=device)
        valid = ~torch.isin(s * n_tgt + t, pos_encoded)
        s_valid, t_valid = s[valid], t[valid]
        needed = n_neg - n_collected
        collected_s.append(s_valid[:needed])
        collected_t.append(t_valid[:needed])
        n_collected += min(len(s_valid), needed)
        if n_collected >= n_neg:
            break

    if not collected_s:
        empty = torch.zeros(0, dtype=torch.long, device=device)
        return empty, empty

    return torch.cat(collected_s)[:n_neg], torch.cat(collected_t)[:n_neg]


def _edge_prediction_loss(
    model: "ShipHGT",
    x_dict: dict[str, Tensor],
    src_type: str,
    tgt_type: str,
    pos_src: Tensor,
    pos_tgt: Tensor,
    all_src: Tensor,
    all_tgt: Tensor,
    n_nodes_src: int,
    n_nodes_tgt: int,
) -> Tensor:
    """BCE loss for masked edge link prediction with balanced negative sampling.

    Parameters
    ----------
    pos_src / pos_tgt:
        Indices of the *masked* (removed) edges — positive examples.
    all_src / all_tgt:
        Indices of *all* edges in this type (kept + masked), used to exclude
        known positives from the negative sample pool.
    """
    n_pos = pos_src.shape[0]
    if n_pos == 0:
        return torch.tensor(0.0, device=pos_src.device)

    device = pos_src.device
    # Encode all positive pairs once as integers for vectorised isin lookup.
    all_encoded = all_src * n_nodes_tgt + all_tgt
    neg_src, neg_tgt = _sample_negative_edges(n_nodes_src, n_nodes_tgt, all_encoded, n_pos, device)
    n_neg = neg_src.shape[0]
    if n_neg == 0:
        return torch.tensor(0.0, device=device)

    combined_src = torch.cat([pos_src, neg_src])
    combined_tgt = torch.cat([pos_tgt, neg_tgt])
    labels = torch.cat([
        torch.ones(n_pos, dtype=torch.float, device=device),
        torch.zeros(n_neg, dtype=torch.float, device=device),
    ])
    logits = model.predict_edge(x_dict, src_type, combined_src, tgt_type, combined_tgt).squeeze(-1)
    return F.binary_cross_entropy_with_logits(logits, labels)


# ---------------------------------------------------------------------------
# Train / eval
# ---------------------------------------------------------------------------

def train_epoch(
    model: "ShipHGT",
    loader: "DataLoader",  # type: ignore[name-defined]
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    mask_rate: float,
    mask_token_idx: int,
    *,
    rotation_mask_rate: float = 0.0,
    overclock_mask_rate: float = 0.0,
    door_mask_rate: float = 0.0,
    virtual_edge_mask_rate: float = 0.0,
    scaler: "torch.cuda.amp.GradScaler | None" = None,
    grad_clip: float = 1.0,
) -> dict[str, float]:
    """Run one training epoch; returns per-task loss/accuracy metrics.

    All loss metrics are averaged per batch so they are on comparable scales
    regardless of how many samples participate in each auxiliary task.
    """
    model.train()
    total_loss = 0.0
    total_correct = 0
    total_masked = 0
    total_part_loss = 0.0
    total_rot_loss = 0.0
    total_rot_correct = 0
    total_rot_masked = 0
    total_oc_loss = 0.0
    total_door_loss = 0.0
    total_virt_loss = 0.0
    skipped_batches = 0
    n_batches = 0

    for batch in loader:
        batch = batch.to(device)
        optimizer.zero_grad()
        n_batches += 1

        # --- Part ID masking (always on) ---
        batch, part_idx, part_labels = apply_part_mask(batch, mask_rate, mask_token_idx)

        # --- Rotation masking (independent node selection) ---
        rot_labels = None
        rot_part_idx: Tensor = part_idx[:0]  # empty sentinel; overwritten below if enabled
        if rotation_mask_rate > 0.0:
            batch, rot_part_idx, rot_labels = apply_rotation_mask(batch, rotation_mask_rate)

        # --- Overclocked masking ---
        oc_idx = oc_labels = None
        if overclock_mask_rate > 0.0:
            batch, oc_idx, oc_labels = apply_overclock_mask(batch, overclock_mask_rate)

        # --- Door edge masking ---
        # Save full edge set before masking so negatives can exclude all real edges.
        door_all_src = door_all_tgt = door_pos_src = door_pos_tgt = None
        if door_mask_rate > 0.0:
            ei = getattr(batch[DOOR_EDGE_KEY], "edge_index", None)
            if ei is not None and ei.shape[1] > 0:
                door_all_src = ei[0].clone()
                door_all_tgt = ei[1].clone()
                batch, door_pos_src, door_pos_tgt = apply_edge_mask(batch, DOOR_EDGE_KEY, door_mask_rate)

        # --- Virtual edge masking ---
        # Each entry: (edge_key, all_src, all_tgt, pos_src, pos_tgt)
        virt_masks: list[tuple[tuple[str, str, str], Tensor, Tensor, Tensor, Tensor]] = []
        if virtual_edge_mask_rate > 0.0:
            for ek in _VIRTUAL_EDGE_TYPES:
                ei = getattr(batch[ek], "edge_index", None)
                if ei is not None and ei.shape[1] > 0:
                    all_src = ei[0].clone()
                    all_tgt = ei[1].clone()
                    batch, pos_src, pos_tgt = apply_edge_mask(batch, ek, virtual_edge_mask_rate)
                    if pos_src.shape[0] > 0:
                        virt_masks.append((ek, all_src, all_tgt, pos_src, pos_tgt))

        # --- Forward pass + loss (AMP autocast when scaler is active) ---
        amp_enabled = scaler is not None and device.type == "cuda"
        with torch.autocast("cuda", enabled=amp_enabled):
            x_dict = model(batch)
            # Virtual dropout may have zeroed some node types; skip edge prediction
            # loss for those types since their embeddings carry no useful signal.
            dropped_virtual = model._last_dropped_virtual_types

            # --- Part ID loss ---
            part_logits = model.predict_parts(x_dict)[part_idx]
            part_loss = F.cross_entropy(part_logits, part_labels)
            loss: Tensor = part_loss
            total_part_loss += part_loss.item()

            # --- Rotation loss (weighted 0.7× to reduce gradient interference) ---
            if rot_labels is not None and rot_labels.shape[0] > 0:
                rot_logits = model.predict_rotation(x_dict, rot_part_idx)
                rot_loss = F.cross_entropy(rot_logits, rot_labels)
                loss = loss + 0.7 * rot_loss
                total_rot_loss += rot_loss.item()
                total_rot_correct += (rot_logits.argmax(dim=1) == rot_labels).sum().item()
                total_rot_masked += rot_labels.shape[0]

            # --- Overclocked loss ---
            if oc_idx is not None and oc_labels is not None and oc_idx.shape[0] > 0:
                oc_logits = model.predict_overclock(x_dict, oc_idx).squeeze(-1)
                # Use pos_weight to compensate for class imbalance (most parts not overclocked).
                n_oc = float(oc_labels.sum().item())
                n_non_oc = float(oc_labels.shape[0]) - n_oc
                pos_weight = torch.tensor([max(1.0, n_non_oc / max(1.0, n_oc))], device=device)
                oc_loss = F.binary_cross_entropy_with_logits(oc_logits, oc_labels, pos_weight=pos_weight)
                loss = loss + oc_loss
                total_oc_loss += oc_loss.item()

            # --- Door edge loss ---
            if door_pos_src is not None and door_pos_src.shape[0] > 0:
                n_parts = batch["part"].num_nodes
                door_loss = _edge_prediction_loss(
                    model, x_dict, "part", "part",
                    door_pos_src, door_pos_tgt,
                    door_all_src, door_all_tgt,
                    n_parts, n_parts,
                )
                loss = loss + door_loss
                total_door_loss += door_loss.item()

            # --- Virtual membership edge loss ---
            if virt_masks:
                virt_loss: Tensor = torch.tensor(0.0, device=device)
                for ek, all_src, all_tgt, pos_src, pos_tgt in virt_masks:
                    src_type, _, tgt_type = ek
                    # Skip if virtual dropout zeroed this node type's features this step.
                    if src_type in dropped_virtual:
                        continue
                    n_src = getattr(batch[src_type], "num_nodes", 0) or 0
                    n_tgt = getattr(batch[tgt_type], "num_nodes", 0) or 0
                    if n_src > 0 and n_tgt > 0:
                        virt_loss = virt_loss + _edge_prediction_loss(
                            model, x_dict, src_type, tgt_type,
                            pos_src, pos_tgt, all_src, all_tgt,
                            n_src, n_tgt,
                        )
                loss = loss + virt_loss
                total_virt_loss += virt_loss.item()

        if not torch.isfinite(loss):
            skipped_batches += 1
            optimizer.zero_grad(set_to_none=True)
            log.warning("Skipping non-finite HGT training batch with loss=%s", loss.detach().item())
            continue

        if scaler is not None:
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            if grad_clip > 0.0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()

        n_masked = len(part_idx)
        total_loss += loss.item()
        total_correct += (part_logits.argmax(dim=1) == part_labels).sum().item()
        total_masked += n_masked

    batch_denom = max(1, n_batches)
    part_denom = max(1, total_masked)
    return {
        "loss": total_loss / batch_denom,
        "acc": total_correct / part_denom,
        "part_loss": total_part_loss / batch_denom,
        "rotation_loss": total_rot_loss / batch_denom,
        "rotation_acc": total_rot_correct / max(1, total_rot_masked),
        "overclock_loss": total_oc_loss / batch_denom,
        "door_loss": total_door_loss / batch_denom,
        "virtual_edge_loss": total_virt_loss / batch_denom,
        "skipped_batches": float(skipped_batches),
        "scaler_scale": float(getattr(scaler, "get_scale", lambda: 1.0)()) if scaler is not None else 1.0,
    }


@torch.no_grad()
def eval_epoch(
    model: "ShipHGT",
    loader: "DataLoader",  # type: ignore[name-defined]
    device: torch.device,
    mask_rate: float,
    mask_token_idx: int,
    *,
    rotation_mask_rate: float = 0.0,
    overclock_mask_rate: float = 0.0,
    door_mask_rate: float = 0.0,
    virtual_edge_mask_rate: float = 0.0,
    amp: bool = False,
) -> dict[str, float]:
    """Run one evaluation pass; returns MLM metrics and any enabled aux metrics."""
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_top5 = 0
    total_masked = 0
    total_rot_loss = 0.0
    total_rot_correct = 0
    total_rot_masked = 0
    total_oc_loss = 0.0
    total_oc_correct = 0
    total_oc_masked = 0
    total_door_loss = 0.0
    total_virt_loss = 0.0
    n_batches = 0

    for batch in loader:
        batch = batch.to(device)
        n_batches += 1
        batch, part_idx, part_labels = apply_part_mask(batch, mask_rate, mask_token_idx)

        rot_labels = None
        rot_part_idx: Tensor = part_idx[:0]
        if rotation_mask_rate > 0.0:
            batch, rot_part_idx, rot_labels = apply_rotation_mask(batch, rotation_mask_rate)

        oc_idx = oc_labels = None
        if overclock_mask_rate > 0.0:
            batch, oc_idx, oc_labels = apply_overclock_mask(batch, overclock_mask_rate)

        door_all_src = door_all_tgt = door_pos_src = door_pos_tgt = None
        if door_mask_rate > 0.0:
            ei = getattr(batch[DOOR_EDGE_KEY], "edge_index", None)
            if ei is not None and ei.shape[1] > 0:
                door_all_src = ei[0].clone()
                door_all_tgt = ei[1].clone()
                batch, door_pos_src, door_pos_tgt = apply_edge_mask(batch, DOOR_EDGE_KEY, door_mask_rate)

        virt_masks: list[tuple[tuple[str, str, str], Tensor, Tensor, Tensor, Tensor]] = []
        if virtual_edge_mask_rate > 0.0:
            for ek in _VIRTUAL_EDGE_TYPES:
                ei = getattr(batch[ek], "edge_index", None)
                if ei is not None and ei.shape[1] > 0:
                    all_src = ei[0].clone()
                    all_tgt = ei[1].clone()
                    batch, pos_src, pos_tgt = apply_edge_mask(batch, ek, virtual_edge_mask_rate)
                    if pos_src.shape[0] > 0:
                        virt_masks.append((ek, all_src, all_tgt, pos_src, pos_tgt))

        with torch.autocast("cuda", enabled=(amp and device.type == "cuda")):
            x_dict = model(batch)
            logits = model.predict_parts(x_dict)
            masked_logits = logits[part_idx]

            loss = F.cross_entropy(masked_logits, part_labels)
            top1 = masked_logits.argmax(dim=1)
            top5 = masked_logits.topk(min(5, masked_logits.size(1)), dim=1).indices

            n_masked = len(part_idx)
            total_loss += loss.item() * n_masked
            total_correct += (top1 == part_labels).sum().item()
            total_top5 += (top5 == part_labels.unsqueeze(1)).any(dim=1).sum().item()
            total_masked += n_masked

            if rot_labels is not None and rot_labels.shape[0] > 0:
                rot_logits = model.predict_rotation(x_dict, rot_part_idx)
                rot_loss = F.cross_entropy(rot_logits, rot_labels)
                total_rot_loss += rot_loss.item()
                total_rot_correct += (rot_logits.argmax(dim=1) == rot_labels).sum().item()
                total_rot_masked += rot_labels.shape[0]

            if oc_idx is not None and oc_labels is not None and oc_idx.shape[0] > 0:
                oc_logits = model.predict_overclock(x_dict, oc_idx).squeeze(-1)
                n_oc = float(oc_labels.sum().item())
                n_non_oc = float(oc_labels.shape[0]) - n_oc
                pos_weight = torch.tensor([max(1.0, n_non_oc / max(1.0, n_oc))], device=device)
                total_oc_loss += F.binary_cross_entropy_with_logits(oc_logits, oc_labels, pos_weight=pos_weight).item()
                total_oc_correct += ((oc_logits > 0) == oc_labels.bool()).sum().item()
                total_oc_masked += oc_idx.shape[0]

            if door_pos_src is not None and door_pos_src.shape[0] > 0:
                n_parts = batch["part"].num_nodes
                total_door_loss += _edge_prediction_loss(
                    model, x_dict, "part", "part",
                    door_pos_src, door_pos_tgt,
                    door_all_src, door_all_tgt,
                    n_parts, n_parts,
                ).item()

            for ek, all_src, all_tgt, pos_src, pos_tgt in virt_masks:
                src_type, _, tgt_type = ek
                n_src = getattr(batch[src_type], "num_nodes", 0) or 0
                n_tgt = getattr(batch[tgt_type], "num_nodes", 0) or 0
                if n_src > 0 and n_tgt > 0:
                    total_virt_loss += _edge_prediction_loss(
                        model, x_dict, src_type, tgt_type,
                        pos_src, pos_tgt, all_src, all_tgt,
                        n_src, n_tgt,
                    ).item()

    part_denom = max(1, total_masked)
    batch_denom = max(1, n_batches)
    return {
        "loss": total_loss / part_denom,
        "acc": total_correct / part_denom,
        "top5_acc": total_top5 / part_denom,
        "rotation_loss": total_rot_loss / batch_denom,
        "rotation_acc": total_rot_correct / max(1, total_rot_masked),
        "overclock_loss": total_oc_loss / batch_denom,
        "overclock_acc": total_oc_correct / max(1, total_oc_masked),
        "door_loss": total_door_loss / batch_denom,
        "virtual_edge_loss": total_virt_loss / batch_denom,
    }


# ---------------------------------------------------------------------------
# Dataset loading
# ---------------------------------------------------------------------------

def load_dataset(pt_paths: list[Path]) -> list["HeteroData"]:
    """Load all ``.pt`` files into memory and return as a list of HeteroData."""
    dataset = []
    errors = 0
    for p in pt_paths:
        try:
            dataset.append(torch.load(p, weights_only=False))
        except Exception as exc:
            log.warning("Failed to load %s: %s", p.name, exc)
            errors += 1
    if errors:
        log.warning("%d .pt files could not be loaded and were skipped", errors)
    return dataset


# ---------------------------------------------------------------------------
# Checkpoint helpers
# ---------------------------------------------------------------------------

def save_checkpoint(
    path: Path,
    model: "ShipHGT",
    optimizer: torch.optim.Optimizer,
    epoch: int,
    metrics: dict[str, float],
    config: dict,
    *,
    scaler: object | None = None,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    checkpoint = {
        "epoch": epoch,
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "metrics": metrics,
        "config": config,
    }
    if scaler is not None:
        checkpoint["scaler_state"] = scaler.state_dict()
    torch.save(checkpoint, path)


def load_checkpoint(
    path: Path,
    model: "ShipHGT",
    optimizer: "torch.optim.Optimizer | None" = None,
    *,
    scaler: object | None = None,
) -> dict:
    checkpoint = torch.load(path, weights_only=False)
    missing, unexpected = model.load_state_dict(checkpoint["model_state"], strict=False)
    if missing:
        log.warning("Checkpoint missing keys (will be randomly initialised): %s", missing)
    if unexpected:
        log.warning("Checkpoint has unexpected keys (ignored): %s", unexpected)
    if optimizer is not None:
        optimizer.load_state_dict(checkpoint["optimizer_state"])
    if scaler is not None and checkpoint.get("scaler_state") is not None:
        scaler.load_state_dict(checkpoint["scaler_state"])
    return checkpoint
