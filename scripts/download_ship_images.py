#!/usr/bin/env python3
"""Download every `.ship.png` attachment visible to the bot in a target Discord guild."""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import socket
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import AsyncIterator, Iterable

import aiohttp
import discord
from dotenv import load_dotenv
from os import getenv

from common.ship_filters import (
    DEFAULT_OPT_OUT_CSV_PATH,
    delete_opted_out_ship_files,
    load_opt_out_author_names,
)

GUILD_ID = 546229904488923141
TOKEN_ENV_VAR = "DISCORD_BOT_TOKEN"
DEFAULT_CHANNEL_ALLOWLIST = [
    546229904488923145,
    546947839008440330,
    546907635149045775,
    1297445027835936779,
    1101149194498089051,
    546329445032787987,
    546327169014431746,
    546327605738209291,
    546333653605679104,
    546333675906662400,
    1318750623302418452,
    561981489357651980,
    1240185912525324300,
    546555689464627212,
]
SAVE_EVERY_MESSAGES = 100
STARTUP_TIMEOUT_SECONDS = 60
STARTUP_MAX_RETRIES = 12
STARTUP_RETRY_DELAY_SECONDS = 10
RETRYABLE_EXCEPTIONS = (
    aiohttp.ClientError,
    discord.HTTPException,
    ConnectionResetError,
    socket.gaierror,
    OSError,
)


def configure_logging(verbose: bool) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)s | %(message)s",
        force=True,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Download every .ship.png attachment from configured text/forum/thread channel IDs."
        )
    )
    parser.add_argument(
        "--output-dir",
        default="downloaded_ships",
        help="Directory where image files will be saved (default: downloaded_ships)",
    )
    parser.add_argument(
        "--channel-id",
        type=int,
        action="append",
        default=[],
        help=(
            "Discord channel/thread ID to scan. Repeat this flag to provide multiple IDs. "
            "If omitted (and --channels-file omitted), the built-in allowlist is used."
        ),
    )
    parser.add_argument(
        "--channels-file",
        help=(
            "Path to a file containing one channel/thread ID per line. "
            "Can be combined with --channel-id."
        ),
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose logging",
    )
    parser.add_argument(
        "--opt-out-csv",
        default=str(DEFAULT_OPT_OUT_CSV_PATH),
        help="CSV containing exact author names to exclude after download",
    )
    return parser.parse_args()


def collision_safe_path(target_dir: Path, filename: str, message_id: int) -> Path:
    candidate = target_dir / filename
    if not candidate.exists():
        return candidate

    stem = candidate.stem
    suffix = candidate.suffix
    return target_dir / f"{stem}__msg{message_id}{suffix}"


def _read_channel_ids_from_file(path: Path) -> list[int]:
    ids: list[int] = []
    try:
        with path.open("r", encoding="utf-8") as f:
            for line_number, raw_line in enumerate(f, start=1):
                line = raw_line.strip()
                if not line or line.startswith("#"):
                    continue
                try:
                    ids.append(int(line))
                except ValueError:
                    logging.warning(
                        "Ignoring invalid channel ID '%s' in %s line %d",
                        line,
                        path,
                        line_number,
                    )
    except OSError as exc:
        logging.warning("Could not read channels file %s: %s", path, exc)
    return ids


def resolve_channel_ids(channel_ids: list[int], channels_file: str | None) -> list[int]:
    resolved: list[int] = []
    seen: set[int] = set()

    def add_ids(values: Iterable[int]) -> None:
        for value in values:
            if value in seen:
                continue
            seen.add(value)
            resolved.append(value)

    add_ids(channel_ids)
    if channels_file:
        add_ids(_read_channel_ids_from_file(Path(channels_file)))

    if not resolved:
        return list(DEFAULT_CHANNEL_ALLOWLIST)
    return resolved


class ShipImageDownloader(discord.Client):
    def __init__(
        self,
        output_dir: Path,
        channel_ids: list[int],
        opt_out_author_names: set[str],
        startup_ready_file: Path | None = None,
        **kwargs: object,
    ) -> None:
        super().__init__(**kwargs)
        self.output_dir = output_dir
        self.guild_id = GUILD_ID
        self.channel_ids = channel_ids
        self.opt_out_author_names = opt_out_author_names
        self.downloaded = 0
        self.filtered = 0
        self.seen_messages = 0
        self.state_path = self.output_dir / "state.json"
        self.state: dict[str, object] = {
            "targets": {},
            "downloaded_filenames": [],
            "processed_attachment_ids": [],
        }
        self.downloaded_filenames: set[str] = set()
        self.processed_attachment_ids: set[str] = set()
        self._messages_since_save = 0
        self.ready_event: asyncio.Event = asyncio.Event()
        self.startup_ready_file = startup_ready_file

    async def on_ready(self) -> None:
        self.ready_event.set()
        if self.startup_ready_file is not None:
            try:
                self.startup_ready_file.write_text("ready\n", encoding="utf-8")
            except OSError as exc:
                logging.warning("Could not write startup ready marker %s: %s", self.startup_ready_file, exc)
        logging.info("Logged in as %s (%s)", self.user, self.user.id if self.user else "unknown")

        guild = self.get_guild(self.guild_id)
        if guild is None:
            logging.error("Guild %s is not available to this bot", self.guild_id)
            await self.close()
            return

        me = guild.me
        if me is None:
            logging.error("Could not resolve bot member record in guild %s", guild.id)
            await self.close()
            return

        self.output_dir.mkdir(parents=True, exist_ok=True)
        self._load_state()
        self._cleanup_existing_opt_out_files()

        scan_targets = await self._collect_scan_targets(guild, me)
        logging.info(
            "Scanning %d target(s) resolved from %d configured channel ID(s) in guild %s",
            len(scan_targets),
            len(self.channel_ids),
            guild.id,
        )

        for idx, target in enumerate(scan_targets, start=1):
            logging.info("[%d/%d] %s", idx, len(scan_targets), target["label"])
            await self._scan_message_source(target)

        self._save_state(force=True)
        logging.info(
            "Completed scan. Messages processed: %d | Matching files downloaded: %d | Filtered by opt-out: %d",
            self.seen_messages,
            self.downloaded,
            self.filtered,
        )
        await self.close()

    async def _scan_message_source(self, target: dict[str, object]) -> None:
        source = target["source"]
        source_name = str(target["name"])
        target_id = int(target["id"])
        scanned_in_channel = 0
        downloaded_in_channel = 0
        retry_count = 0

        while True:
            after_obj = self._after_for_target(target_id)
            try:
                async for message in source.history(limit=None, oldest_first=True, after=after_obj):
                    retry_count = 0
                    scanned_in_channel += 1
                    self.seen_messages += 1

                    for attachment in message.attachments:
                        filename = attachment.filename
                        if not filename.lower().endswith(".ship.png"):
                            continue

                        attachment_key = str(attachment.id)
                        if attachment_key in self.processed_attachment_ids:
                            continue

                        output_path = collision_safe_path(self.output_dir, filename, message.id)
                        output_name = output_path.name
                        if output_name in self.downloaded_filenames:
                            self._track_processed_attachment(attachment_key)
                            continue

                        download_result = await self._save_attachment_with_retries(
                            attachment,
                            output_path,
                            message_id=message.id,
                            source_name=source_name,
                        )
                        if download_result == "downloaded":
                            downloaded_in_channel += 1
                            self._record_retained_download(attachment_key, output_name)
                        elif download_result == "filtered":
                            self.filtered += 1

                    self._update_target_progress(target_id, message.id)

                    if scanned_in_channel % 500 == 0:
                        logging.info(
                            "Progress #%s: %d messages scanned, %d downloads",
                            source_name,
                            scanned_in_channel,
                            downloaded_in_channel,
                        )

                logging.info(
                    "Finished #%s: %d messages scanned, %d downloads",
                    source_name,
                    scanned_in_channel,
                    downloaded_in_channel,
                )
                return
            except discord.Forbidden:
                logging.warning("Missing permission to read history for #%s", source_name)
                return
            except RETRYABLE_EXCEPTIONS as exc:
                retry_count += 1
                self._save_state(force=True)
                wait_seconds = min(60, 2 ** min(retry_count, 6))
                logging.warning(
                    "Retryable error while scanning #%s (attempt %d): %s. Backing off for %ss",
                    source_name,
                    retry_count,
                    exc,
                    wait_seconds,
                )
                await asyncio.sleep(wait_seconds)

    async def _save_attachment_with_retries(
        self,
        attachment: discord.Attachment,
        output_path: Path,
        message_id: int,
        source_name: str,
    ) -> str:
        """Download one attachment and delete it immediately when it matches the opt-out list."""

        retry_count = 0
        while True:
            try:
                await attachment.save(output_path, use_cached=False)
                if self._filter_downloaded_ship_file(output_path):
                    self._save_state(force=True)
                    return "filtered"
                logging.info(
                    "Downloaded %s from message %s in #%s",
                    output_path.name,
                    message_id,
                    source_name,
                )
                self._save_state(force=True)
                return "downloaded"
            except RETRYABLE_EXCEPTIONS as exc:
                retry_count += 1
                self._save_state(force=True)
                wait_seconds = min(60, 2 ** min(retry_count, 6))
                logging.warning(
                    "Retryable download failure for %s (message %s, attempt %d): %s. Backing off for %ss",
                    attachment.url,
                    message_id,
                    retry_count,
                    exc,
                    wait_seconds,
                )
                await asyncio.sleep(wait_seconds)
            except Exception as exc:  # noqa: BLE001
                logging.warning(
                    "Failed to download attachment %s from message %s: %s",
                    attachment.url,
                    message_id,
                    exc,
                )
                return "failed"

    async def _collect_scan_targets(
        self,
        guild: discord.Guild,
        me: discord.Member,
    ) -> list[dict[str, object]]:
        targets: list[dict[str, object]] = []
        seen_target_ids: set[int] = set()

        def add_target(target: dict[str, object]) -> None:
            target_id = int(target["id"])
            if target_id in seen_target_ids:
                return
            seen_target_ids.add(target_id)
            targets.append(target)

        for channel_id in self.channel_ids:
            channel = guild.get_channel(channel_id)
            if channel is None:
                channel = guild.get_thread(channel_id)

            if channel is None:
                logging.warning("Configured channel ID %s was not found or is not accessible", channel_id)
                continue

            if isinstance(channel, discord.TextChannel):
                if not self._can_read_channel(channel, me):
                    logging.warning("Skipping inaccessible text channel #%s (%s)", channel.name, channel.id)
                    continue

                add_target(
                    {
                        "id": channel.id,
                        "source": channel,
                        "name": channel.name,
                        "label": f"Text channel #{channel.name} ({channel.id})",
                    }
                )

                for thread in self._filter_accessible_threads(channel.threads, me):
                    add_target(self._thread_target(thread, f"Thread in #{channel.name}"))

                async for thread in self._iter_archived_threads(channel):
                    if not self._can_read_thread(thread, me):
                        continue
                    add_target(self._thread_target(thread, f"Archived thread in #{channel.name}"))

                continue

            if isinstance(channel, discord.ForumChannel):
                if not self._can_read_channel(channel, me):
                    logging.warning("Skipping inaccessible forum channel #%s (%s)", channel.name, channel.id)
                    continue

                for thread in self._filter_accessible_threads(channel.threads, me):
                    add_target(self._thread_target(thread, f"Forum thread in #{channel.name}"))

                async for thread in self._iter_archived_threads(channel):
                    if not self._can_read_thread(thread, me):
                        continue
                    add_target(self._thread_target(thread, f"Archived forum thread in #{channel.name}"))

                continue

            if isinstance(channel, discord.Thread):
                if not self._can_read_thread(channel, me):
                    logging.warning("Skipping inaccessible thread #%s (%s)", channel.name, channel.id)
                    continue
                add_target(self._thread_target(channel, "Thread"))
                continue

            logging.warning(
                "Skipping unsupported channel type for ID %s: %s",
                channel_id,
                type(channel).__name__,
            )

        return targets

    def _thread_target(self, thread: discord.Thread, prefix: str) -> dict[str, object]:
        return {
            "id": thread.id,
            "source": thread,
            "name": thread.name,
            "label": f"{prefix} #{thread.name} ({thread.id})",
        }

    def _filter_accessible_threads(
        self,
        threads: Iterable[discord.Thread],
        me: discord.Member,
    ) -> list[discord.Thread]:
        return [thread for thread in threads if self._can_read_thread(thread, me)]

    def _can_read_thread(self, thread: discord.Thread, me: discord.Member) -> bool:
        perms = thread.permissions_for(me)
        return perms.view_channel and perms.read_message_history

    def _can_read_channel(
        self,
        channel: discord.TextChannel | discord.ForumChannel,
        me: discord.Member,
    ) -> bool:
        perms = channel.permissions_for(me)
        return perms.view_channel and perms.read_message_history

    async def _iter_archived_threads(
        self,
        channel: discord.TextChannel | discord.ForumChannel,
    ) -> AsyncIterator[discord.Thread]:
        if isinstance(channel, discord.TextChannel):
            try:
                async for thread in channel.archived_threads(limit=None, private=False):
                    yield thread
            except discord.Forbidden:
                logging.warning("Missing permission to list archived threads for #%s", channel.name)
            except RETRYABLE_EXCEPTIONS as exc:
                logging.warning(
                    "Error listing archived threads for #%s: %s",
                    channel.name,
                    exc,
                )

            try:
                async for thread in channel.archived_threads(limit=None, private=True, joined=True):
                    yield thread
            except discord.Forbidden:
                logging.warning(
                    "Missing permission to list archived private threads for #%s",
                    channel.name,
                )
            except RETRYABLE_EXCEPTIONS as exc:
                logging.warning(
                    "Error listing archived private threads for #%s: %s",
                    channel.name,
                    exc,
                )
            return

        try:
            async for thread in channel.archived_threads(limit=None):
                yield thread
        except discord.Forbidden:
            logging.warning("Missing permission to list archived threads for #%s", channel.name)
        except RETRYABLE_EXCEPTIONS as exc:
            logging.warning("Error listing archived threads for #%s: %s", channel.name, exc)

    def _load_state(self) -> None:
        if not self.state_path.exists():
            self.state = {
                "targets": {},
                "downloaded_filenames": [],
                "processed_attachment_ids": [],
            }
            return

        try:
            data = json.loads(self.state_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError) as exc:
            logging.warning("Could not read state file %s: %s. Starting fresh.", self.state_path, exc)
            self.state = {
                "targets": {},
                "downloaded_filenames": [],
                "processed_attachment_ids": [],
            }
            return

        targets = data.get("targets", {})
        filenames = data.get("downloaded_filenames", [])
        attachment_ids = data.get(
            "processed_attachment_ids",
            data.get("downloaded_attachment_ids", []),
        )

        if not isinstance(targets, dict):
            targets = {}
        if not isinstance(filenames, list):
            filenames = []
        if not isinstance(attachment_ids, list):
            attachment_ids = []

        self.state = {
            "targets": targets,
            "downloaded_filenames": filenames,
            "processed_attachment_ids": attachment_ids,
        }
        self.downloaded_filenames = {str(item) for item in filenames}
        self.processed_attachment_ids = {str(item) for item in attachment_ids}
        logging.info(
            "Loaded resume state: %d target checkpoint(s), %d tracked filename(s), %d processed attachment(s)",
            len(targets),
            len(self.downloaded_filenames),
            len(self.processed_attachment_ids),
        )

    def _save_state(self, force: bool = False) -> None:
        if not force and self._messages_since_save < SAVE_EVERY_MESSAGES:
            return

        self._messages_since_save = 0
        payload = {
            "targets": self.state.get("targets", {}),
            "downloaded_filenames": sorted(self.downloaded_filenames),
            # Keep writing the legacy key so older script versions can still resume cleanly
            "downloaded_attachment_ids": sorted(self.processed_attachment_ids),
            "processed_attachment_ids": sorted(self.processed_attachment_ids),
        }

        tmp_path = self.state_path.with_suffix(".json.tmp")
        try:
            tmp_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
            tmp_path.replace(self.state_path)
        except OSError as exc:
            logging.warning("Failed to persist state file %s: %s", self.state_path, exc)

    def _update_target_progress(self, target_id: int, message_id: int) -> None:
        targets = self.state.setdefault("targets", {})
        if isinstance(targets, dict):
            targets[str(target_id)] = {"last_message_id": message_id}

        self._messages_since_save += 1
        self._save_state()

    def _after_for_target(self, target_id: int) -> discord.Object | None:
        targets = self.state.get("targets", {})
        if not isinstance(targets, dict):
            return None

        target_state = targets.get(str(target_id), {})
        if not isinstance(target_state, dict):
            return None

        last_message_id = target_state.get("last_message_id")
        if not isinstance(last_message_id, int) or last_message_id <= 0:
            return None

        return discord.Object(id=last_message_id)

    def _track_processed_attachment(self, attachment_key: str) -> None:
        """Track an attachment ID whose downloaded file should be kept long term."""

        self.processed_attachment_ids.add(attachment_key)

    def _track_downloaded_filename(self, output_name: str) -> None:
        """Track one filename that currently exists in the download directory."""

        self.downloaded_filenames.add(output_name)

    def _record_retained_download(self, attachment_key: str, output_name: str) -> None:
        """Record one retained download so future scans can skip it safely."""

        self.downloaded += 1
        self._track_downloaded_filename(output_name)
        self._track_processed_attachment(attachment_key)

    def _cleanup_existing_opt_out_files(self) -> None:
        """Delete any already-downloaded ships that now match the opt-out list."""

        filter_summary = delete_opted_out_ship_files([self.output_dir], self.opt_out_author_names)
        if filter_summary["ship_files_deleted"] == 0:
            return

        deleted_filenames = {
            Path(path_text).name for path_text in filter_summary["deleted_ship_paths"]
        }
        self.downloaded_filenames.difference_update(deleted_filenames)
        logging.info(
            "Removed %d existing opted-out ship file(s) from %s",
            filter_summary["ship_files_deleted"],
            self.output_dir,
        )
        self._save_state(force=True)

    def _filter_downloaded_ship_file(self, output_path: Path) -> bool:
        """Delete one just-downloaded ship file when its author is opted out."""

        filter_summary = delete_opted_out_ship_files([output_path], self.opt_out_author_names)
        if filter_summary["ship_files_deleted"] == 0:
            return False

        # Keep the in-memory filename set aligned with what still exists on disk
        self.downloaded_filenames.discard(output_path.name)
        return True


def _run_single_download(
    output_dir: str,
    channel_ids: list[int],
    token: str,
    verbose: bool,
    opt_out_csv: str | Path = DEFAULT_OPT_OUT_CSV_PATH,
    startup_ready_file: Path | None = None,
) -> int:
    configure_logging(verbose)
    opt_out_author_names = load_opt_out_author_names(opt_out_csv)
    intents = discord.Intents.none()
    intents.guilds = True
    intents.messages = True
    client = ShipImageDownloader(
        output_dir=Path(output_dir),
        channel_ids=channel_ids,
        opt_out_author_names=opt_out_author_names,
        startup_ready_file=startup_ready_file,
        intents=intents,
    )
    try:
        client.run(token)
    except discord.LoginFailure as exc:
        logging.error("Discord login failed: %s", exc)
        return 1
    except Exception as exc:  # noqa: BLE001
        logging.error("Discord client exited with error: %s", exc)
        return 1
    return 0


def run_download(
    output_dir: str | Path = "downloaded_ships",
    verbose: bool = False,
    channel_ids: list[int] | None = None,
    channels_file: str | None = None,
    opt_out_csv: str | Path = DEFAULT_OPT_OUT_CSV_PATH,
    startup_timeout_seconds: int = STARTUP_TIMEOUT_SECONDS,
) -> int:
    configure_logging(verbose)

    load_dotenv()
    token = getenv(TOKEN_ENV_VAR)
    if not token:
        logging.error("Missing %s in .env or environment", TOKEN_ENV_VAR)
        return 1

    resolved_channel_ids = resolve_channel_ids(channel_ids or [], channels_file)
    logging.info("Configured %d channel ID(s) for scanning", len(resolved_channel_ids))

    resolved_output_dir = str(Path(output_dir))
    resolved_opt_out_csv = str(Path(opt_out_csv))

    for attempt in range(1, STARTUP_MAX_RETRIES + 1):
        ready_fd, ready_path_str = tempfile.mkstemp(prefix="ship_download_ready_", suffix=".tmp")
        os.close(ready_fd)
        ready_path = Path(ready_path_str)
        ready_path.unlink(missing_ok=True)

        child_env = os.environ.copy()
        child_env["SHIP_DOWNLOAD_WORKER"] = "1"
        child_env["SHIP_DOWNLOAD_OUTPUT_DIR"] = resolved_output_dir
        child_env["SHIP_DOWNLOAD_CHANNEL_IDS"] = json.dumps(resolved_channel_ids)
        child_env["SHIP_DOWNLOAD_READY_FILE"] = str(ready_path)
        child_env["SHIP_DOWNLOAD_VERBOSE"] = "1" if verbose else "0"
        child_env["SHIP_DOWNLOAD_OPT_OUT_CSV"] = resolved_opt_out_csv
        child_env[TOKEN_ENV_VAR] = token

        process = subprocess.Popen(  # noqa: S603
            [sys.executable, str(Path(__file__).resolve())],
            env=child_env,
        )
        start_ts = time.monotonic()
        while True:
            if ready_path.exists():
                break
            if process.poll() is not None:
                break
            if time.monotonic() - start_ts >= startup_timeout_seconds:
                break
            time.sleep(1)

        timed_out = not ready_path.exists() and process.poll() is None
        if timed_out:
            logging.warning(
                "Download attempt %d/%d timed out waiting %ss for Discord ready; retrying",
                attempt,
                STARTUP_MAX_RETRIES,
                startup_timeout_seconds,
            )
            process.terminate()
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=5)
            ready_path.unlink(missing_ok=True)
            if attempt < STARTUP_MAX_RETRIES:
                time.sleep(STARTUP_RETRY_DELAY_SECONDS)
                continue
            logging.error("All startup attempts failed due timeout")
            return 1

        result = process.wait()
        ready_path.unlink(missing_ok=True)
        if result != 0:
            return result
        return 0

    return 1


def main() -> int:
    if getenv("SHIP_DOWNLOAD_WORKER") == "1":
        output_dir = getenv("SHIP_DOWNLOAD_OUTPUT_DIR", "downloaded_ships")
        raw_channel_ids = getenv("SHIP_DOWNLOAD_CHANNEL_IDS", "[]")
        ready_file_value = getenv("SHIP_DOWNLOAD_READY_FILE")
        verbose_value = getenv("SHIP_DOWNLOAD_VERBOSE", "0") == "1"
        opt_out_csv_value = getenv("SHIP_DOWNLOAD_OPT_OUT_CSV", str(DEFAULT_OPT_OUT_CSV_PATH))
        token = getenv(TOKEN_ENV_VAR)
        if not token:
            logging.error("Missing %s in worker environment", TOKEN_ENV_VAR)
            return 1
        try:
            channel_ids = json.loads(raw_channel_ids)
        except json.JSONDecodeError:
            logging.error("Invalid worker channel list payload")
            return 1
        if not isinstance(channel_ids, list) or not all(isinstance(v, int) for v in channel_ids):
            logging.error("Worker channel list is malformed")
            return 1
        ready_file = Path(ready_file_value) if ready_file_value else None
        return _run_single_download(
            output_dir=output_dir,
            channel_ids=channel_ids,
            token=token,
            verbose=verbose_value,
            opt_out_csv=opt_out_csv_value,
            startup_ready_file=ready_file,
        )

    args = parse_args()
    return run_download(
        output_dir=args.output_dir,
        verbose=args.verbose,
        channel_ids=args.channel_id,
        channels_file=args.channels_file,
        opt_out_csv=args.opt_out_csv,
    )


if __name__ == "__main__":
    raise SystemExit(main())
