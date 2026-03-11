"""Tests for download-time ship opt-out filtering."""

from __future__ import annotations

import asyncio
import importlib.util
import json
from pathlib import Path

import discord

from common.cosmoteer import create_ship_png_bytes


def _load_download_module():
    """Load the download script as a testable Python module."""

    script_path = Path(__file__).resolve().parents[1] / "scripts" / "download_ship_images.py"
    spec = importlib.util.spec_from_file_location("download_ship_images_test", script_path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _build_ship_payload(*, name: str, author: str) -> dict:
    """Build a minimal valid ship payload for downloader tests."""

    return {
        "Version": 1,
        "Name": name,
        "Author": author,
        "FlightDirection": 0,
        "Parts": [
            {
                "ID": "cosmoteer.corridor",
                "Location": [0, 0],
                "Rotation": 0,
            }
        ],
        "Doors": [],
    }


def _write_ship_png(path: Path, *, name: str, author: str) -> None:
    """Write one `.ship.png` fixture to disk."""

    path.write_bytes(create_ship_png_bytes(_build_ship_payload(name=name, author=author)))


class _FakeAttachment:
    """Minimal attachment test double for download-scan state tests."""

    def __init__(self, attachment_id: int, filename: str) -> None:
        """Store just the fields the downloader reads during scanning."""

        self.id = attachment_id
        self.filename = filename
        self.url = f"https://example.invalid/{filename}"


class _FakeMessage:
    """Minimal message test double exposing attachments and an ID."""

    def __init__(self, message_id: int, attachments: list[_FakeAttachment]) -> None:
        """Store the message ID and attachment list for one scan iteration."""

        self.id = message_id
        self.attachments = attachments


class _FakeHistorySource:
    """History source test double that yields a fixed set of messages."""

    def __init__(self, messages: list[_FakeMessage]) -> None:
        """Store message fixtures for later async iteration."""

        self._messages = messages

    async def history(self, limit=None, oldest_first=True, after=None):  # noqa: ANN001
        """Yield stored messages through the same async API shape Discord uses."""

        del limit, oldest_first, after
        for message in self._messages:
            yield message


def test_cleanup_existing_opt_out_files_removes_deleted_names_from_state(tmp_path: Path) -> None:
    """Startup cleanup should delete opted-out files and untrack their filenames."""

    download_module = _load_download_module()
    output_dir = tmp_path / "downloaded_ships"
    blocked_ship_path = output_dir / "blocked.ship.png"
    allowed_ship_path = output_dir / "allowed.ship.png"
    output_dir.mkdir()
    _write_ship_png(blocked_ship_path, name="Blocked", author="blocked")
    _write_ship_png(allowed_ship_path, name="Allowed", author="allowed")

    downloader = download_module.ShipImageDownloader(
        output_dir=output_dir,
        channel_ids=[],
        opt_out_author_names={"blocked"},
        intents=discord.Intents.none(),
    )
    downloader.downloaded_filenames = {
        blocked_ship_path.name,
        allowed_ship_path.name,
    }

    downloader._cleanup_existing_opt_out_files()

    assert not blocked_ship_path.exists()
    assert allowed_ship_path.exists()
    assert blocked_ship_path.name not in downloader.downloaded_filenames
    assert allowed_ship_path.name in downloader.downloaded_filenames


def test_filter_downloaded_ship_file_deletes_matching_author(tmp_path: Path) -> None:
    """A just-downloaded file should be removed immediately when author matches."""

    download_module = _load_download_module()
    output_dir = tmp_path / "downloaded_ships"
    blocked_ship_path = output_dir / "blocked.ship.png"
    output_dir.mkdir()
    _write_ship_png(blocked_ship_path, name="Blocked", author="blocked")

    downloader = download_module.ShipImageDownloader(
        output_dir=output_dir,
        channel_ids=[],
        opt_out_author_names={"blocked"},
        intents=discord.Intents.none(),
    )

    assert downloader._filter_downloaded_ship_file(blocked_ship_path) is True
    assert not blocked_ship_path.exists()


def test_filtered_attachments_are_not_persisted_as_processed(tmp_path: Path) -> None:
    """Filtered attachments should not be permanently skipped in future runs."""

    download_module = _load_download_module()
    output_dir = tmp_path / "downloaded_ships"
    output_dir.mkdir()

    downloader = download_module.ShipImageDownloader(
        output_dir=output_dir,
        channel_ids=[],
        opt_out_author_names={"blocked"},
        intents=discord.Intents.none(),
    )

    async def _fake_save_attachment_with_retries(*args, **kwargs) -> str:  # noqa: ANN002, ANN003
        """Pretend the downloaded attachment was filtered out by the opt-out step."""

        del args, kwargs
        return "filtered"

    downloader._save_attachment_with_retries = _fake_save_attachment_with_retries
    target = {
        "id": 123,
        "name": "test-channel",
        "source": _FakeHistorySource(
            [
                _FakeMessage(
                    456,
                    [_FakeAttachment(789, "blocked.ship.png")],
                )
            ]
        ),
    }

    asyncio.run(downloader._scan_message_source(target))
    downloader._save_state(force=True)
    persisted_state = json.loads(downloader.state_path.read_text(encoding="utf-8"))

    assert downloader.filtered == 1
    assert downloader.processed_attachment_ids == set()
    assert persisted_state["processed_attachment_ids"] == []
