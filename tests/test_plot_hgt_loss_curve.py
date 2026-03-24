from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

SCRIPT_PATH = Path(__file__).resolve().parent.parent / "scripts" / "plot_hgt_loss_curve.py"
_SPEC = importlib.util.spec_from_file_location("plot_hgt_loss_curve", SCRIPT_PATH)
assert _SPEC is not None and _SPEC.loader is not None
_MODULE = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = _MODULE
_SPEC.loader.exec_module(_MODULE)

main = _MODULE.main
parse_loss_log = _MODULE.parse_loss_log


def test_parse_loss_log_ignores_non_epoch_lines(tmp_path) -> None:
    log_path = tmp_path / "train_log.txt"
    log_path.write_text(
        "\n".join(
            [
                "starting run",
                "epoch 001/010 train_loss=2.0 val_loss=1.5 train_acc=0.25",
                "epoch 002/010 train_loss=1.4 val_loss=1.1 train_acc=0.40",
                "epoch 003/010 train_loss=nan val_loss=0.9 train_acc=0.50",
            ]
        ),
        encoding="utf-8",
    )

    points = parse_loss_log(log_path)

    assert [(point.epoch, point.train_loss, point.val_loss) for point in points] == [
        (1, 2.0, 1.5),
        (2, 1.4, 1.1),
    ]


def test_main_derives_output_from_run_folder(tmp_path, monkeypatch, capsys) -> None:
    monkeypatch.chdir(tmp_path)

    log_path = Path("models/experiment-42/train_log.txt")
    log_path.parent.mkdir(parents=True)
    log_path.write_text("epoch 001/010 train_loss=2.0 val_loss=1.5 train_acc=0.25\n", encoding="utf-8")

    exit_code = main(["--input", str(log_path)])

    assert exit_code == 0
    output_path = Path("out/visualizations/loss-curves/experiment-42-loss-curve.png")
    assert output_path.exists()
    assert capsys.readouterr().out.strip() == str(output_path)
