from __future__ import annotations

import argparse
import math
from types import SimpleNamespace

import torch

import training.cli
from training.backends.hgt.backend import HGTTrainingBackend, _resolve_virtual_mask_rates
from training.backends.hgt import train as train_module
from training.backends.hgt.train import load_checkpoint, save_checkpoint, train_epoch


class _FakeScaler:
    def __init__(self, state: dict[str, float]) -> None:
        self._state = dict(state)

    def state_dict(self) -> dict[str, float]:
        return dict(self._state)

    def load_state_dict(self, state: dict[str, float]) -> None:
        self._state = dict(state)


class _TrainScaler:
    def __init__(self, scale: float = 8.0) -> None:
        self._scale = scale
        self.step_calls = 0

    def scale(self, loss: torch.Tensor):
        class _ScaledLoss:
            def __init__(self, inner: torch.Tensor) -> None:
                self._inner = inner

            def backward(self) -> None:
                self._inner.backward()

        return _ScaledLoss(loss)

    def step(self, optimizer: torch.optim.Optimizer) -> None:
        self.step_calls += 1
        optimizer.step()

    def unscale_(self, optimizer: torch.optim.Optimizer) -> None:
        del optimizer

    def update(self, new_scale: float | None = None) -> None:
        if new_scale is not None:
            self._scale = float(new_scale)

    def get_scale(self) -> float:
        return self._scale


class _FakeBatch:
    def __init__(self) -> None:
        self.part = SimpleNamespace(
            num_nodes=2,
            part_id=torch.tensor([0, 1], dtype=torch.long),
            x=torch.zeros((2, 7), dtype=torch.float),
        )

    def to(self, device: torch.device) -> "_FakeBatch":
        self.part.part_id = self.part.part_id.to(device)
        self.part.x = self.part.x.to(device)
        return self

    def __getitem__(self, key: str):
        if key == "part":
            return self.part
        raise KeyError(key)


class _TinyModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.proj = torch.nn.Linear(7, 4)
        self.head = torch.nn.Linear(4, 3)
        self._last_dropped_virtual_types = frozenset()

    def forward(self, batch: _FakeBatch) -> dict[str, torch.Tensor]:
        return {"part": self.proj(batch["part"].x)}

    def predict_parts(self, x_dict: dict[str, torch.Tensor]) -> torch.Tensor:
        return self.head(x_dict["part"])

    def predict_rotation(
        self,
        x_dict: dict[str, torch.Tensor],
        mask_indices: torch.Tensor,
    ) -> torch.Tensor:
        return x_dict["part"][mask_indices]


def test_hgt_checkpoint_roundtrips_scaler_state(tmp_path) -> None:
    model = torch.nn.Linear(4, 2)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    scaler = _FakeScaler({"scale": 4096.0, "growth_tracker": 7.0})

    checkpoint_path = tmp_path / "amp-checkpoint.pt"
    save_checkpoint(
        checkpoint_path,
        model,
        optimizer,
        epoch=3,
        metrics={"loss": 1.25},
        config={"hidden_dim": 8},
        scaler=scaler,
    )

    restored_model = torch.nn.Linear(4, 2)
    restored_optimizer = torch.optim.AdamW(restored_model.parameters(), lr=1e-3)
    restored_scaler = _FakeScaler({"scale": 1.0, "growth_tracker": 0.0})

    checkpoint = load_checkpoint(
        checkpoint_path,
        restored_model,
        restored_optimizer,
        scaler=restored_scaler,
    )

    assert checkpoint["epoch"] == 3
    assert checkpoint["scaler_state"] == {"scale": 4096.0, "growth_tracker": 7.0}
    assert restored_scaler.state_dict() == {"scale": 4096.0, "growth_tracker": 7.0}


def test_hgt_validate_parser_accepts_amp_flag() -> None:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command")
    backend_subparsers = subparsers.add_parser("validate").add_subparsers(dest="backend")

    HGTTrainingBackend().register_validate_parser(backend_subparsers)

    args = parser.parse_args(
        [
            "validate",
            "hgt",
            "--checkpoint",
            "models/hgt/best.pt",
            "--input-dir",
            "graphs",
            "--vocab",
            "models/hgt/vocab.json",
            "--amp",
        ]
    )

    assert args.amp is True


def test_hgt_stats_parser_is_registered() -> None:
    parser = training.cli.build_parser()
    args = parser.parse_args(
        [
            "stats",
            "hgt",
            "--input-dir",
            "graphs",
        ]
    )
    assert args.action == "stats"
    assert args.backend == "hgt"


def test_resolve_virtual_mask_rates_uses_legacy_default() -> None:
    args = argparse.Namespace(
        virtual_edge_mask_rate=0.3,
        virtual_edge_mask_rate_dense=None,
        virtual_edge_mask_rate_sparse=None,
    )
    dense, sparse = _resolve_virtual_mask_rates(args)
    assert dense == 0.3
    assert sparse == 0.3


def test_resolve_virtual_mask_rates_prefers_split_values() -> None:
    args = argparse.Namespace(
        virtual_edge_mask_rate=0.3,
        virtual_edge_mask_rate_dense=0.2,
        virtual_edge_mask_rate_sparse=0.6,
    )
    dense, sparse = _resolve_virtual_mask_rates(args)
    assert dense == 0.2
    assert sparse == 0.6


def test_hgt_train_epoch_skips_nonfinite_loss_batches(monkeypatch) -> None:
    model = _TinyModel()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    scaler = _TrainScaler(scale=8.0)
    loader = [_FakeBatch(), _FakeBatch()]

    def _fake_apply_part_mask(batch, mask_rate, mask_token_idx):
        del mask_rate, mask_token_idx
        return batch, torch.tensor([0], dtype=torch.long), torch.tensor([1], dtype=torch.long)

    cross_entropy_calls = 0
    original_cross_entropy = train_module.F.cross_entropy

    def _fake_cross_entropy(logits, labels):
        nonlocal cross_entropy_calls
        cross_entropy_calls += 1
        if cross_entropy_calls == 1:
            return torch.tensor(float("nan"), device=logits.device, requires_grad=True)
        return original_cross_entropy(logits, labels)

    monkeypatch.setattr(train_module, "apply_part_mask", _fake_apply_part_mask)
    monkeypatch.setattr(train_module.F, "cross_entropy", _fake_cross_entropy)

    metrics = train_epoch(
        model,
        loader,
        optimizer,
        torch.device("cpu"),
        mask_rate=0.15,
        mask_token_idx=2,
        scaler=scaler,
    )

    assert metrics["skipped_batches"] == 1.0
    assert math.isfinite(metrics["loss"])
    assert metrics["loss"] > 0.0
    assert scaler.step_calls == 1
    assert metrics["scaler_scale"] == 8.0


def test_hgt_train_epoch_rotation_acc_uses_masked_sample_count(monkeypatch) -> None:
    model = _TinyModel()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    loader = [_FakeBatch()]

    def _fake_apply_part_mask(batch, mask_rate, mask_token_idx):
        del mask_rate, mask_token_idx
        return batch, torch.tensor([0, 1], dtype=torch.long), torch.tensor([1, 2], dtype=torch.long)

    def _fake_apply_rotation_mask(batch, mask_rate):
        del batch
        assert mask_rate == 0.075
        return loader[0], torch.tensor([0], dtype=torch.long), torch.tensor([3], dtype=torch.long)

    def _fake_predict_rotation(_x_dict, mask_indices):
        assert mask_indices.numel() == 1
        return torch.tensor([[0.0, 0.0, 0.0, 1.0]], dtype=torch.float, requires_grad=True)

    monkeypatch.setattr(train_module, "apply_part_mask", _fake_apply_part_mask)
    monkeypatch.setattr(train_module, "apply_rotation_mask", _fake_apply_rotation_mask)
    monkeypatch.setattr(model, "predict_rotation", _fake_predict_rotation)

    metrics = train_epoch(
        model,
        loader,
        optimizer,
        torch.device("cpu"),
        mask_rate=0.15,
        mask_token_idx=2,
        rotation_mask_rate=0.075,
    )

    assert metrics["rotation_acc"] == 1.0
