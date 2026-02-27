#!/usr/bin/env python3
"""Download every `.ship.png` attachment visible to the bot in a target Discord guild."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

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
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download every .ship.png attachment from all accessible text channels."
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

        channels: list[discord.TextChannel] = []
        for channel in guild.text_channels:
            perms = channel.permissions_for(me)
            if perms.view_channel and perms.read_message_history:
                channels.append(channel)

        logging.info("Scanning %d accessible text channel(s) in guild %s", len(channels), guild.id)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        for idx, channel in enumerate(channels, start=1):
            logging.info("[%d/%d] Channel #%s (%s)", idx, len(channels), channel.name, channel.id)
            await self._scan_channel(channel)

        logging.info(
            "Completed scan. Messages processed: %d | Matching files downloaded: %d",
            self.seen_messages,
            self.downloaded,
        )
        await self.close()

    async def _scan_channel(self, channel: discord.TextChannel) -> None:
        scanned_in_channel = 0
        downloaded_in_channel = 0

        try:
            async for message in channel.history(limit=None, oldest_first=True):
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
                            channel.name,
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
                        channel.name,
                        scanned_in_channel,
                        downloaded_in_channel,
                    )

            logging.info(
                "Finished #%s: %d messages scanned, %d downloads",
                channel.name,
                scanned_in_channel,
                downloaded_in_channel,
            )
        except discord.Forbidden:
            logging.warning("Missing permission to read history for #%s", channel.name)
        except discord.HTTPException as exc:
            logging.warning("HTTP error while scanning #%s: %s", channel.name, exc)


def main() -> int:
    args = parse_args()
    configure_logging(args.verbose)

    load_dotenv()
    token = getenv(TOKEN_ENV_VAR)
    if not token:
        logging.error("Missing %s in .env or environment", TOKEN_ENV_VAR)
        return 1

    intents = discord.Intents.none()
    intents.guilds = True
    intents.messages = True

    client = ShipImageDownloader(output_dir=Path(args.output_dir), intents=intents)
    client.run(token)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
