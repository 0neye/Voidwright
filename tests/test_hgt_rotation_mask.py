from __future__ import annotations

import torch

from training.backends.hgt.model import ROTATION_MASK_IDX, ShipHGT
from training.backends.hgt.train import OVERCLOCK_MASK_VALUE, apply_overclock_mask, apply_rotation_mask


class _RotationBatch:
    def __init__(self) -> None:
        self.part = type(
            "PartStore",
            (),
            {
                "num_nodes": 4,
                "rotation": torch.tensor([0, 1, 2, 3], dtype=torch.long),
            },
        )()

    def __getitem__(self, key: str):
        if key == "part":
            return self.part
        raise KeyError(key)


class _OverclockBatch:
    def __init__(self) -> None:
        self.part = type(
            "PartStore",
            (),
            {
                "num_nodes": 4,
                "x": torch.tensor(
                    [
                        [0.0, 0.0, 1.0, 1.0, 1.0, 0.0, 0.0],
                        [0.0, 0.0, 1.0, 1.0, 1.0, 0.0, 1.0],
                        [0.0, 0.0, 1.0, 1.0, 1.0, 0.0, 0.0],
                        [0.0, 0.0, 1.0, 1.0, 1.0, 0.0, 1.0],
                    ],
                    dtype=torch.float,
                ),
            },
        )()

    def __getitem__(self, key: str):
        if key == "part":
            return self.part
        raise KeyError(key)


def test_apply_rotation_mask_uses_dedicated_mask_idx(monkeypatch) -> None:
    batch = _RotationBatch()

    monkeypatch.setattr(torch, "randperm", lambda n, device=None: torch.tensor([0, 2, 1, 3], device=device))
    _, indices, labels = apply_rotation_mask(batch, 0.5)

    assert indices.tolist() == [0, 2]
    assert labels.tolist() == [0, 2]
    assert batch["part"].rotation.tolist() == [ROTATION_MASK_IDX, 1, ROTATION_MASK_IDX, 3]


def test_ship_hgt_allocates_rotation_mask_embedding_slot() -> None:
    model = ShipHGT(vocab_size=8)

    assert model.rotation_embed.num_embeddings == ROTATION_MASK_IDX + 1


def test_apply_overclock_mask_uses_dedicated_mask_value(monkeypatch) -> None:
    batch = _OverclockBatch()

    monkeypatch.setattr(torch, "randperm", lambda n, device=None: torch.tensor([1, 3, 0, 2], device=device))
    _, indices, labels = apply_overclock_mask(batch, mask_rate=0.5)

    assert indices.tolist() == [1, 3]
    assert labels.tolist() == [1.0, 1.0]
    assert batch["part"].x[:, 6].tolist() == [0.0, OVERCLOCK_MASK_VALUE, 0.0, OVERCLOCK_MASK_VALUE]
