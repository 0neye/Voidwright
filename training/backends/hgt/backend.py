"""HGT backend adapter for the generic training module."""

from __future__ import annotations

import argparse
import logging
import random
from pathlib import Path

import orjson
import torch

from training.backends.hgt.stats import collect_corpus_stats, rotation_class_weights_from_stats
from training.base import TrainingBackend

__all__ = ["HGTTrainingBackend"]

log = logging.getLogger(__name__)


def _resolve_virtual_mask_rates(args: argparse.Namespace) -> tuple[float, float]:
    """Resolve dense/sparse virtual mask rates with legacy flag compatibility."""
    legacy = args.virtual_edge_mask_rate
    dense = args.virtual_edge_mask_rate_dense
    sparse = args.virtual_edge_mask_rate_sparse
    if legacy is not None and dense is None:
        dense = legacy
    if legacy is not None and sparse is None:
        sparse = legacy
    return float(dense or 0.0), float(sparse or 0.0)


def _parse_rotation_weights_arg(value: str) -> torch.Tensor | None:
    """Parse rotation class-weight CLI argument."""
    lowered = value.strip().lower()
    if lowered == "none":
        return None
    if lowered == "auto":
        raise ValueError("'auto' requires corpus stats and cannot be parsed directly")
    parts = [part.strip() for part in value.split(",") if part.strip()]
    if len(parts) != 4:
        raise ValueError("rotation class weights must have exactly four comma-separated values")
    try:
        parsed = [float(part) for part in parts]
    except ValueError as exc:
        raise ValueError("rotation class weights must be floats") from exc
    return torch.tensor(parsed, dtype=torch.float)


class HGTTrainingBackend(TrainingBackend):
    """Train a Heterogeneous Graph Transformer encoder on expanded ship graphs."""

    name = "hgt"

    def register_stats_parser(self, backend_subparsers: argparse._SubParsersAction) -> None:
        parser = backend_subparsers.add_parser(
            self.name,
            help="Compute corpus statistics used to calibrate HGT masking/loss settings",
        )
        parser.add_argument(
            "--input-dir",
            type=Path,
            required=True,
            metavar="DIR",
            help="Directory containing expanded ship graph JSON files",
        )
        parser.add_argument(
            "--output",
            type=Path,
            default=Path("models/hgt/corpus-stats.json"),
            help="Path to write the corpus stats JSON report",
        )

    def register_build_parser(self, backend_subparsers: argparse._SubParsersAction) -> None:
        parser = backend_subparsers.add_parser(
            self.name,
            help="Train an HGT encoder on expanded ship graph corpora",
        )
        # Data
        parser.add_argument(
            "--input-dir",
            type=Path,
            required=True,
            metavar="DIR",
            help="Directory containing expanded (and optionally filtered) ship graph JSON files",
        )
        parser.add_argument(
            "--output-dir",
            type=Path,
            default=Path("models/hgt"),
            metavar="DIR",
            help="Directory to write vocab.json, best checkpoint, and training log",
        )
        parser.add_argument(
            "--cache-dir",
            type=Path,
            default=None,
            metavar="DIR",
            help="Directory for converted .pt cache files (default: --output-dir/cache)",
        )
        parser.add_argument(
            "--force-reconvert",
            action="store_true",
            help="Reconvert all graph files even if cached .pt files exist",
        )
        parser.add_argument(
            "--resume",
            type=Path,
            default=None,
            metavar="CHECKPOINT",
            help="Resume training from this checkpoint (.pt); skips completed epochs",
        )
        # Model
        parser.add_argument("--hidden-dim", type=int, default=128)
        parser.add_argument("--num-heads", type=int, default=4)
        parser.add_argument("--num-layers", type=int, default=2)
        parser.add_argument("--dropout", type=float, default=0.1)
        parser.add_argument("--pe-dim", type=int, default=32,
                            help="Sinusoidal positional encoding dimension (must be div by 4)")
        parser.add_argument("--edge-features", action="store_true", default=False,
                            help="Use EdgeAwareHGTConv with per-edge feature attention bias (shared_sides, travel_distance)")
        # Training
        parser.add_argument("--epochs", type=int, default=50)
        parser.add_argument("--batch-size", type=int, default=8,
                            help="Number of graphs per batch")
        parser.add_argument("--lr", type=float, default=1e-3)
        parser.add_argument("--weight-decay", type=float, default=1e-2)
        parser.add_argument("--mask-rate", type=float, default=0.15,
                            help="Fraction of part nodes to mask per batch")
        parser.add_argument("--virtual-dropout", type=float, default=0.3,
                            help="Probability of zeroing each virtual node type per forward pass")
        parser.add_argument("--rotation-mask-rate", type=float, default=0.0,
                            help="Fraction of masked part nodes whose rotation is also masked and predicted (0 = disabled)")
        parser.add_argument(
            "--rotation-class-weights",
            type=str,
            default="auto",
            help="Rotation CE class weights: auto, none, or comma-separated w0,w1,w2,w3",
        )
        parser.add_argument("--overclock-mask-rate", type=float, default=0.15,
                            help="Overclock masking budget. Targeted masking always includes OC positives, then pads with non-OC nodes")
        parser.add_argument("--door-mask-rate", type=float, default=0.15,
                            help="Fraction of door edges removed before message passing for link prediction")
        parser.add_argument(
            "--virtual-edge-mask-rate",
            type=float,
            default=None,
            help="Legacy single virtual-edge mask rate. If set and split rates are omitted, applies to dense and sparse groups",
        )
        parser.add_argument("--virtual-edge-mask-rate-dense", type=float, default=None,
                            help="Mask rate for dense virtual memberships (cluster)")
        parser.add_argument("--virtual-edge-mask-rate-sparse", type=float, default=None,
                            help="Mask rate for sparse virtual memberships (thermal/weapon_grp)")
        parser.add_argument("--no-reverse-edges", action="store_true",
                            help="Disable reverse membership edges (part → virtual)")
        parser.add_argument("--val-split", type=float, default=0.1,
                            help="Fraction of graphs held out for validation")
        parser.add_argument("--seed", type=int, default=42)
        parser.add_argument(
            "--device",
            type=str,
            default=None,
            help="Compute device (e.g. 'cuda', 'cpu'). Defaults to cuda if available.",
        )
        parser.add_argument(
            "--amp",
            action="store_true",
            default=False,
            help="Enable automatic mixed precision (fp16) training. Reduces GPU memory usage.",
        )

    def register_validate_parser(self, backend_subparsers: argparse._SubParsersAction) -> None:
        parser = backend_subparsers.add_parser(
            self.name,
            help="Evaluate a trained HGT checkpoint on masked part prediction",
        )
        parser.add_argument(
            "--checkpoint",
            type=Path,
            required=True,
            help="Path to a saved HGT checkpoint (.pt)",
        )
        parser.add_argument(
            "--input-dir",
            type=Path,
            required=True,
            metavar="DIR",
            help="Directory containing expanded ship graph JSON files to evaluate on",
        )
        parser.add_argument(
            "--vocab",
            type=Path,
            required=True,
            help="Path to vocab.json produced during training",
        )
        parser.add_argument(
            "--output",
            type=Path,
            default=Path("models/hgt/validation.json"),
            help="Path to write the validation report JSON",
        )
        parser.add_argument(
            "--cache-dir",
            type=Path,
            default=None,
            metavar="DIR",
        )
        parser.add_argument("--batch-size", type=int, default=8)
        parser.add_argument("--mask-rate", type=float, default=0.15)
        parser.add_argument("--rotation-mask-rate", type=float, default=0.0)
        parser.add_argument(
            "--rotation-class-weights",
            type=str,
            default="auto",
            help="Rotation CE class weights: auto, none, or comma-separated w0,w1,w2,w3",
        )
        parser.add_argument("--overclock-mask-rate", type=float, default=0.15)
        parser.add_argument("--door-mask-rate", type=float, default=0.15)
        parser.add_argument(
            "--virtual-edge-mask-rate",
            type=float,
            default=None,
            help="Legacy single virtual-edge mask rate. If set and split rates are omitted, applies to dense and sparse groups",
        )
        parser.add_argument("--virtual-edge-mask-rate-dense", type=float, default=None)
        parser.add_argument("--virtual-edge-mask-rate-sparse", type=float, default=None)
        parser.add_argument("--no-reverse-edges", action="store_true")
        parser.add_argument("--seed", type=int, default=42)
        parser.add_argument("--device", type=str, default=None)
        parser.add_argument(
            "--amp",
            action="store_true",
            default=False,
            help="Enable automatic mixed precision (fp16) evaluation on CUDA.",
        )

    # ------------------------------------------------------------------
    # Build
    # ------------------------------------------------------------------

    def run_build(self, args: argparse.Namespace) -> int:
        from torch_geometric.loader import DataLoader

        from training.backends.hgt.convert import convert_corpus
        from training.backends.hgt.model import ShipHGT
        from training.backends.hgt.train import (
            eval_epoch,
            load_checkpoint,
            load_dataset,
            save_checkpoint,
            train_epoch,
        )
        from training.backends.hgt.vocab import VocabRegistry

        prefix = "[training:hgt]"

        # Device
        if args.device:
            device = torch.device(args.device)
        else:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"{prefix} using device: {device}")

        # Reproducibility
        torch.manual_seed(args.seed)
        random.seed(args.seed)

        args.output_dir.mkdir(parents=True, exist_ok=True)
        cache_dir = args.cache_dir or (args.output_dir / "cache")
        dense_virtual_rate, sparse_virtual_rate = _resolve_virtual_mask_rates(args)

        # Vocab
        vocab_path = args.output_dir / "vocab.json"
        if vocab_path.exists() and not args.force_reconvert:
            print(f"{prefix} loading existing vocab from {vocab_path}")
            vocab = VocabRegistry.load(vocab_path)
        else:
            print(f"{prefix} building vocab from {args.input_dir} ...")
            vocab = VocabRegistry.build_from_corpus(args.input_dir)
            vocab.save(vocab_path)
        print(f"{prefix} vocab size: {len(vocab)} ({vocab.num_classes} predictable classes)")

        # Convert corpus to .pt cache
        print(f"{prefix} converting graphs to {cache_dir} ...")
        reverse_edges = not args.no_reverse_edges
        pt_paths = convert_corpus(
            args.input_dir, cache_dir, vocab, force=args.force_reconvert,
            reverse_edges=reverse_edges,
        )
        if not pt_paths:
            print(f"{prefix} ERROR: no graphs found in {args.input_dir}")
            return 1
        print(f"{prefix} {len(pt_paths)} graphs available")

        # Corpus stats drive automatic calibration for weighted rotation loss.
        corpus_stats = collect_corpus_stats(args.input_dir)
        corpus_stats_path = args.output_dir / "corpus-stats.json"
        corpus_stats_path.write_bytes(orjson.dumps(corpus_stats, option=orjson.OPT_INDENT_2))
        print(f"{prefix} corpus stats written to {corpus_stats_path}")
        rotation_weights_arg = args.rotation_class_weights.strip().lower()
        if rotation_weights_arg == "auto":
            rotation_class_weights = rotation_class_weights_from_stats(corpus_stats)
        else:
            try:
                rotation_class_weights = _parse_rotation_weights_arg(args.rotation_class_weights)
            except ValueError as exc:
                print(f"{prefix} ERROR: {exc}")
                return 1
        if rotation_class_weights is None:
            print(f"{prefix} rotation CE class weights: disabled")
        else:
            weights_str = ",".join(f"{w:.4f}" for w in rotation_class_weights.tolist())
            print(f"{prefix} rotation CE class weights: [{weights_str}]")
        print(
            f"{prefix} virtual edge mask rates: dense={dense_virtual_rate:.3f}, sparse={sparse_virtual_rate:.3f}"
        )

        # Load and split
        random.shuffle(pt_paths)
        n_val = max(1, int(len(pt_paths) * args.val_split))
        val_paths, train_paths = pt_paths[:n_val], pt_paths[n_val:]
        print(f"{prefix} train: {len(train_paths)}, val: {len(val_paths)}")

        train_data = load_dataset(train_paths)
        val_data = load_dataset(val_paths)
        if not train_data:
            print(f"{prefix} ERROR: no training graphs loaded")
            return 1

        train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True)
        val_loader = DataLoader(val_data, batch_size=args.batch_size, shuffle=False)

        # Model
        model_config = {
            "vocab_size": len(vocab),
            "hidden_dim": args.hidden_dim,
            "num_heads": args.num_heads,
            "num_layers": args.num_layers,
            "dropout": args.dropout,
            "pe_dim": args.pe_dim,
            "virtual_dropout_rate": args.virtual_dropout,
            "reverse_edges": reverse_edges,
            "edge_features": args.edge_features,
        }
        model = ShipHGT(**model_config).to(device)
        n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"{prefix} model parameters: {n_params:,}")

        optimizer = torch.optim.AdamW(
            model.parameters(), lr=args.lr, weight_decay=args.weight_decay
        )

        use_amp = args.amp and device.type == "cuda"
        scaler = torch.amp.GradScaler("cuda") if use_amp else None
        if use_amp:
            print(f"{prefix} AMP enabled (fp16 autocast)")

        # Resume from checkpoint if requested.
        start_epoch = 1
        best_val_loss = float("inf")
        if args.resume:
            print(f"{prefix} resuming from {args.resume}")
            ckpt = load_checkpoint(args.resume, model, optimizer, scaler=scaler)
            start_epoch = ckpt["epoch"] + 1
            if ckpt.get("metrics") and "loss" in ckpt["metrics"]:
                best_val_loss = ckpt["metrics"]["loss"]
            print(f"{prefix} continuing from epoch {start_epoch}, best_val_loss={best_val_loss:.4f}")

        # last_epoch=-1 on fresh start; ckpt["epoch"]-1 when resuming so the
        # cosine schedule continues from where it left off.
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=args.epochs, last_epoch=start_epoch - 2
        )

        best_ckpt = args.output_dir / "best.pt"
        val_metrics: dict[str, float] = {}
        log_path = args.output_dir / "train_log.txt"
        # Open in append mode so resumed runs extend the existing log.
        log_file = log_path.open("a", encoding="utf-8")

        for epoch in range(start_epoch, args.epochs + 1):
            train_metrics = train_epoch(
                model, train_loader, optimizer, device, args.mask_rate, vocab.mask_idx,
                rotation_mask_rate=args.rotation_mask_rate,
                rotation_class_weights=rotation_class_weights,
                overclock_mask_rate=args.overclock_mask_rate,
                door_mask_rate=args.door_mask_rate,
                virtual_edge_mask_rate_dense=dense_virtual_rate,
                virtual_edge_mask_rate_sparse=sparse_virtual_rate,
                scaler=scaler,
            )
            val_metrics = eval_epoch(
                model, val_loader, device, args.mask_rate, vocab.mask_idx,
                rotation_mask_rate=args.rotation_mask_rate,
                rotation_class_weights=rotation_class_weights,
                overclock_mask_rate=args.overclock_mask_rate,
                door_mask_rate=args.door_mask_rate,
                virtual_edge_mask_rate_dense=dense_virtual_rate,
                virtual_edge_mask_rate_sparse=sparse_virtual_rate,
                amp=use_amp,
            )
            scheduler.step()

            aux = []
            if args.rotation_mask_rate > 0.0:
                aux.append(f"rot_loss={train_metrics['rotation_loss']:.4f}/{val_metrics['rotation_loss']:.4f}")
                aux.append(f"rot_acc={val_metrics['rotation_acc']:.4f}")
            if args.overclock_mask_rate > 0.0:
                aux.append(f"oc_loss={train_metrics['overclock_loss']:.4f}/{val_metrics['overclock_loss']:.4f}")
                aux.append(f"oc_acc={val_metrics['overclock_acc']:.4f}")
                aux.append(f"oc_recall={val_metrics['overclock_recall']:.4f}")
            if args.door_mask_rate > 0.0:
                aux.append(f"door_loss={train_metrics['door_loss']:.4f}/{val_metrics['door_loss']:.4f}")
            if dense_virtual_rate > 0.0 or sparse_virtual_rate > 0.0:
                aux.append(
                    "virt_loss(d/s)="
                    f"{train_metrics['virtual_edge_loss_dense']:.4f}/{train_metrics['virtual_edge_loss_sparse']:.4f}"
                    f" vs {val_metrics['virtual_edge_loss_dense']:.4f}/{val_metrics['virtual_edge_loss_sparse']:.4f}"
                )
                aux.append(
                    "virt_acc(d/s)="
                    f"{val_metrics['virtual_edge_acc_dense']:.4f}/{val_metrics['virtual_edge_acc_sparse']:.4f}"
                )
            if train_metrics["skipped_batches"] > 0.0:
                aux.append(f"skipped_batches={int(train_metrics['skipped_batches'])}")
            if use_amp:
                aux.append(f"amp_scale={train_metrics['scaler_scale']:.1f}")
            aux_str = ("  " + "  ".join(aux)) if aux else ""
            line = (
                f"epoch {epoch:03d}/{args.epochs}  "
                f"train_loss={train_metrics['loss']:.4f}  train_acc={train_metrics['acc']:.4f}  "
                f"val_loss={val_metrics['loss']:.4f}  val_acc={val_metrics['acc']:.4f}  "
                f"val_top5={val_metrics['top5_acc']:.4f}"
                f"{aux_str}"
            )
            print(f"{prefix} {line}", flush=True)
            log_file.write(line + "\n")
            log_file.flush()

            if val_metrics["loss"] < best_val_loss:
                best_val_loss = val_metrics["loss"]
                save_checkpoint(
                    best_ckpt,
                    model,
                    optimizer,
                    epoch,
                    val_metrics,
                    model_config,
                    scaler=scaler,
                )
                print(f"{prefix}   → new best checkpoint saved", flush=True)

        log_file.close()

        # Save final checkpoint
        save_checkpoint(
            args.output_dir / "last.pt",
            model,
            optimizer,
            args.epochs,
            val_metrics,
            model_config,
            scaler=scaler,
        )
        print(f"{prefix} training complete. Best val_loss={best_val_loss:.4f}")
        print(f"{prefix} outputs written to {args.output_dir}")
        return 0

    # ------------------------------------------------------------------
    # Validate
    # ------------------------------------------------------------------

    def run_validate(self, args: argparse.Namespace) -> int:
        from torch_geometric.loader import DataLoader

        from training.backends.hgt.convert import convert_corpus
        from training.backends.hgt.model import ShipHGT
        from training.backends.hgt.train import eval_epoch, load_dataset, load_checkpoint
        from training.backends.hgt.vocab import VocabRegistry

        prefix = "[training:hgt:validate]"

        if args.device:
            device = torch.device(args.device)
        else:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        use_amp = args.amp and device.type == "cuda"

        torch.manual_seed(args.seed)
        dense_virtual_rate, sparse_virtual_rate = _resolve_virtual_mask_rates(args)

        rotation_weights_arg = args.rotation_class_weights.strip().lower()
        if rotation_weights_arg == "auto":
            corpus_stats = collect_corpus_stats(args.input_dir)
            rotation_class_weights = rotation_class_weights_from_stats(corpus_stats)
        else:
            try:
                rotation_class_weights = _parse_rotation_weights_arg(args.rotation_class_weights)
            except ValueError as exc:
                print(f"{prefix} ERROR: {exc}")
                return 1

        vocab = VocabRegistry.load(args.vocab)

        cache_dir = args.cache_dir or (args.checkpoint.parent / "val_cache")
        reverse_edges = not args.no_reverse_edges
        pt_paths = convert_corpus(args.input_dir, cache_dir, vocab, force=False,
                                  reverse_edges=reverse_edges)
        if not pt_paths:
            print(f"{prefix} ERROR: no graphs found in {args.input_dir}")
            return 1

        dataset = load_dataset(pt_paths)
        loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)

        # Reconstruct model from checkpoint config
        checkpoint = torch.load(args.checkpoint, weights_only=False)
        config = checkpoint["config"]
        model = ShipHGT(**config).to(device)
        load_checkpoint(args.checkpoint, model)

        metrics = eval_epoch(
            model, loader, device, args.mask_rate, vocab.mask_idx,
            rotation_mask_rate=args.rotation_mask_rate,
            rotation_class_weights=rotation_class_weights,
            overclock_mask_rate=args.overclock_mask_rate,
            door_mask_rate=args.door_mask_rate,
            virtual_edge_mask_rate_dense=dense_virtual_rate,
            virtual_edge_mask_rate_sparse=sparse_virtual_rate,
            amp=use_amp,
        )
        print(f"{prefix} loss={metrics['loss']:.4f}  acc={metrics['acc']:.4f}  top5={metrics['top5_acc']:.4f}")

        report = {
            "checkpoint": str(args.checkpoint),
            "input_dir": str(args.input_dir),
            "num_graphs": len(dataset),
            "mask_rate": args.mask_rate,
            "rotation_class_weights": None if rotation_class_weights is None else rotation_class_weights.tolist(),
            "virtual_edge_mask_rate_dense": dense_virtual_rate,
            "virtual_edge_mask_rate_sparse": sparse_virtual_rate,
            **metrics,
        }
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_bytes(orjson.dumps(report, option=orjson.OPT_INDENT_2))
        print(f"{prefix} report written to {args.output}")
        return 0

    # ------------------------------------------------------------------
    # Stats
    # ------------------------------------------------------------------

    def run_stats(self, args: argparse.Namespace) -> int:
        prefix = "[training:hgt:stats]"
        try:
            payload = collect_corpus_stats(args.input_dir)
        except Exception as exc:
            print(f"{prefix} ERROR: failed to compute stats from {args.input_dir}: {exc}")
            return 1

        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_bytes(orjson.dumps(payload, option=orjson.OPT_INDENT_2))
        print(f"{prefix} report written to {args.output}")
        return 0
