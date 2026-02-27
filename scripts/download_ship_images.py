#!/usr/bin/env python3
"""Download every `.ship.png` attachment visible to the bot in a target Discord guild."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import AsyncIterator, Iterable

import discord
from dotenv import load_dotenv
from os import getenv

GUILD_ID = 546229904488923141
TOKEN_ENV_VAR = "DISCORD_BOT_TOKEN"


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
            "Download every .ship.png attachment from accessible text channels, threads, and forums."
        )
    )
    parser.add_argument(
        "--output-dir",
        default="downloaded_ships",
        help="Directory where image files will be saved (default: downloaded_ships)",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose logging",
    )
    return parser.parse_args()


def collision_safe_path(target_dir: Path, filename: str, message_id: int) -> Path:
    candidate = target_dir / filename
    if not candidate.exists():
        return candidate

    stem = candidate.stem
    suffix = candidate.suffix
    return target_dir / f"{stem}__msg{message_id}{suffix}"


class ShipImageDownloader(discord.Client):
    def __init__(self, output_dir: Path, **kwargs: object) -> None:
        super().__init__(**kwargs)
        self.output_dir = output_dir
        self.guild_id = GUILD_ID
        self.downloaded = 0
        self.seen_messages = 0

    async def on_ready(self) -> None:
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

        text_channels: list[discord.TextChannel] = []
        for channel in guild.text_channels:
            perms = channel.permissions_for(me)
            if perms.view_channel and perms.read_message_history:
                text_channels.append(channel)

        forum_channels: list[discord.ForumChannel] = []
        for forum in guild.forums:
            perms = forum.permissions_for(me)
            if perms.view_channel and perms.read_message_history:
                forum_channels.append(forum)

        scan_targets = await self._collect_scan_targets(text_channels, forum_channels, me)
        logging.info(
            "Scanning %d target(s) in guild %s (%d text channel(s), %d forum channel(s), threads included)",
            len(scan_targets),
            guild.id,
            len(text_channels),
            len(forum_channels),
        )
        self.output_dir.mkdir(parents=True, exist_ok=True)

        for idx, target in enumerate(scan_targets, start=1):
            logging.info("[%d/%d] %s", idx, len(scan_targets), target["label"])
            await self._scan_message_source(target["source"], target["name"])

        logging.info(
            "Completed scan. Messages processed: %d | Matching files downloaded: %d",
            self.seen_messages,
            self.downloaded,
        )
        await self.close()

    async def _scan_message_source(
        self,
        source: discord.abc.Messageable,
        source_name: str,
    ) -> None:
        scanned_in_channel = 0
        downloaded_in_channel = 0

        try:
            async for message in source.history(limit=None, oldest_first=True):
                scanned_in_channel += 1
                self.seen_messages += 1

                for attachment in message.attachments:
                    filename = attachment.filename
                    if not filename.lower().endswith(".ship.png"):
                        continue

                    output_path = collision_safe_path(self.output_dir, filename, message.id)
                    try:
                        await attachment.save(output_path, use_cached=False)
                        self.downloaded += 1
                        downloaded_in_channel += 1
                        logging.info(
                            "Downloaded %s from message %s in #%s",
                            output_path.name,
                            message.id,
                            source_name,
                        )
                    except discord.HTTPException as exc:
                        logging.warning(
                            "Failed to download attachment %s from message %s: %s",
                            attachment.url,
                            message.id,
                            exc,
                        )

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
        except discord.Forbidden:
            logging.warning("Missing permission to read history for #%s", source_name)
        except discord.HTTPException as exc:
            logging.warning("HTTP error while scanning #%s: %s", source_name, exc)

    async def _collect_scan_targets(
        self,
        text_channels: list[discord.TextChannel],
        forum_channels: list[discord.ForumChannel],
        me: discord.Member,
    ) -> list[dict[str, object]]:
        targets: list[dict[str, object]] = []
        seen_thread_ids: set[int] = set()

        for channel in text_channels:
            targets.append(
                {
                    "source": channel,
                    "name": channel.name,
                    "label": f"Text channel #{channel.name} ({channel.id})",
                }
            )

            for thread in self._filter_accessible_threads(channel.threads, me):
                if thread.id in seen_thread_ids:
                    continue
                seen_thread_ids.add(thread.id)
                targets.append(self._thread_target(thread, f"Thread in #{channel.name}"))

            async for thread in self._iter_archived_threads(channel):
                if thread.id in seen_thread_ids:
                    continue
                if not self._can_read_thread(thread, me):
                    continue
                seen_thread_ids.add(thread.id)
                targets.append(self._thread_target(thread, f"Archived thread in #{channel.name}"))

        for forum in forum_channels:
            for thread in self._filter_accessible_threads(forum.threads, me):
                if thread.id in seen_thread_ids:
                    continue
                seen_thread_ids.add(thread.id)
                targets.append(self._thread_target(thread, f"Forum thread in #{forum.name}"))

            async for thread in self._iter_archived_threads(forum):
                if thread.id in seen_thread_ids:
                    continue
                if not self._can_read_thread(thread, me):
                    continue
                seen_thread_ids.add(thread.id)
                targets.append(self._thread_target(thread, f"Archived forum thread in #{forum.name}"))

        return targets

    def _thread_target(self, thread: discord.Thread, prefix: str) -> dict[str, object]:
        return {
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
            except discord.HTTPException as exc:
                logging.warning(
                    "HTTP error listing archived threads for #%s: %s",
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
            except discord.HTTPException as exc:
                logging.warning(
                    "HTTP error listing archived private threads for #%s: %s",
                    channel.name,
                    exc,
                )
            return

        try:
            async for thread in channel.archived_threads(limit=None):
                yield thread
        except discord.Forbidden:
            logging.warning("Missing permission to list archived threads for #%s", channel.name)
        except discord.HTTPException as exc:
            logging.warning("HTTP error listing archived threads for #%s: %s", channel.name, exc)


def run_download(output_dir: str | Path = "downloaded_ships", verbose: bool = False) -> int:
    configure_logging(verbose)

    load_dotenv()
    token = getenv(TOKEN_ENV_VAR)
    if not token:
        logging.error("Missing %s in .env or environment", TOKEN_ENV_VAR)
        return 1

    intents = discord.Intents.none()
    intents.guilds = True
    intents.messages = True

    client = ShipImageDownloader(output_dir=Path(output_dir), intents=intents)
    client.run(token)
    return 0


def main() -> int:
    args = parse_args()
    return run_download(output_dir=args.output_dir, verbose=args.verbose)


if __name__ == "__main__":
    raise SystemExit(main())
