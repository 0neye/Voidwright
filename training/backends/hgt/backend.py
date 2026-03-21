"""HGT backend adapter for the generic training module."""

from __future__ import annotations

import argparse
import logging
import random
from pathlib import Path

from training.base import TrainingBackend

__all__ = ["HGTTrainingBackend"]

log = logging.getLogger(__name__)


class HGTTrainingBackend(TrainingBackend):
    """Train a Heterogeneous Graph Transformer encoder on expanded ship graphs."""

    name = "hgt"

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
        # Model
        parser.add_argument("--hidden-dim", type=int, default=128)
        parser.add_argument("--num-heads", type=int, default=4)
        parser.add_argument("--num-layers", type=int, default=3)
        parser.add_argument("--dropout", type=float, default=0.1)
        parser.add_argument("--pe-dim", type=int, default=32,
                            help="Sinusoidal positional encoding dimension (must be div by 4)")
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
        parser.add_argument("--overclock-mask-rate", type=float, default=0.15,
                            help="Fraction of part nodes whose overclocked flag is masked")
        parser.add_argument("--door-mask-rate", type=float, default=0.15,
                            help="Fraction of door edges removed before message passing for link prediction")
        parser.add_argument("--virtual-edge-mask-rate", type=float, default=0.0,
                            help="Fraction of virtual membership edges removed for link prediction (0 = disabled)")
        parser.add_argument("--val-split", type=float, default=0.1,
                            help="Fraction of graphs held out for validation")
        parser.add_argument("--seed", type=int, default=42)
        parser.add_argument(
            "--device",
            type=str,
            default=None,
            help="Compute device (e.g. 'cuda', 'cpu'). Defaults to cuda if available.",
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
        parser.add_argument("--overclock-mask-rate", type=float, default=0.15)
        parser.add_argument("--door-mask-rate", type=float, default=0.15)
        parser.add_argument("--virtual-edge-mask-rate", type=float, default=0.0)
        parser.add_argument("--seed", type=int, default=42)
        parser.add_argument("--device", type=str, default=None)

    # ------------------------------------------------------------------
    # Build
    # ------------------------------------------------------------------

    def run_build(self, args: argparse.Namespace) -> int:
        import torch
        from torch_geometric.loader import DataLoader

        from training.backends.hgt.convert import convert_corpus
        from training.backends.hgt.model import ShipHGT
        from training.backends.hgt.train import (
            eval_epoch,
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
        pt_paths = convert_corpus(
            args.input_dir, cache_dir, vocab, force=args.force_reconvert
        )
        if not pt_paths:
            print(f"{prefix} ERROR: no graphs found in {args.input_dir}")
            return 1
        print(f"{prefix} {len(pt_paths)} graphs available")

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
        }
        model = ShipHGT(**model_config).to(device)
        n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"{prefix} model parameters: {n_params:,}")

        optimizer = torch.optim.AdamW(
            model.parameters(), lr=args.lr, weight_decay=args.weight_decay
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

        # Training loop
        best_val_loss = float("inf")
        best_ckpt = args.output_dir / "best.pt"
        log_lines: list[str] = []

        for epoch in range(1, args.epochs + 1):
            train_metrics = train_epoch(
                model, train_loader, optimizer, device, args.mask_rate, vocab.mask_idx,
                overclock_mask_rate=args.overclock_mask_rate,
                door_mask_rate=args.door_mask_rate,
                virtual_edge_mask_rate=args.virtual_edge_mask_rate,
            )
            val_metrics = eval_epoch(
                model, val_loader, device, args.mask_rate, vocab.mask_idx,
                overclock_mask_rate=args.overclock_mask_rate,
                door_mask_rate=args.door_mask_rate,
                virtual_edge_mask_rate=args.virtual_edge_mask_rate,
            )
            scheduler.step()

            aux = []
            if args.overclock_mask_rate > 0.0:
                aux.append(f"oc_loss={train_metrics['overclock_loss']:.4f}")
            if args.door_mask_rate > 0.0:
                aux.append(f"door_loss={train_metrics['door_loss']:.4f}")
            if args.virtual_edge_mask_rate > 0.0:
                aux.append(f"virt_loss={train_metrics['virtual_edge_loss']:.4f}")
            aux_str = ("  " + "  ".join(aux)) if aux else ""
            line = (
                f"epoch {epoch:03d}/{args.epochs}  "
                f"train_loss={train_metrics['loss']:.4f}  train_acc={train_metrics['acc']:.4f}  "
                f"val_loss={val_metrics['loss']:.4f}  val_acc={val_metrics['acc']:.4f}  "
                f"val_top5={val_metrics['top5_acc']:.4f}"
                f"{aux_str}"
            )
            print(f"{prefix} {line}")
            log_lines.append(line)

            if val_metrics["loss"] < best_val_loss:
                best_val_loss = val_metrics["loss"]
                save_checkpoint(best_ckpt, model, optimizer, epoch, val_metrics, model_config)
                print(f"{prefix}   → new best checkpoint saved")

        # Save final checkpoint and training log
        save_checkpoint(
            args.output_dir / "last.pt", model, optimizer, args.epochs, val_metrics, model_config
        )
        (args.output_dir / "train_log.txt").write_text("\n".join(log_lines) + "\n", encoding="utf-8")
        print(f"{prefix} training complete. Best val_loss={best_val_loss:.4f}")
        print(f"{prefix} outputs written to {args.output_dir}")
        return 0

    # ------------------------------------------------------------------
    # Validate
    # ------------------------------------------------------------------

    def run_validate(self, args: argparse.Namespace) -> int:
        import torch
        import orjson
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

        torch.manual_seed(args.seed)

        vocab = VocabRegistry.load(args.vocab)

        cache_dir = args.cache_dir or (args.checkpoint.parent / "val_cache")
        pt_paths = convert_corpus(args.input_dir, cache_dir, vocab, force=False)
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
            overclock_mask_rate=args.overclock_mask_rate,
            door_mask_rate=args.door_mask_rate,
            virtual_edge_mask_rate=args.virtual_edge_mask_rate,
        )
        print(f"{prefix} loss={metrics['loss']:.4f}  acc={metrics['acc']:.4f}  top5={metrics['top5_acc']:.4f}")

        report = {
            "checkpoint": str(args.checkpoint),
            "input_dir": str(args.input_dir),
            "num_graphs": len(dataset),
            "mask_rate": args.mask_rate,
            **metrics,
        }
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_bytes(orjson.dumps(report, option=orjson.OPT_INDENT_2))
        print(f"{prefix} report written to {args.output}")
        return 0
