"""Canonicalize and deduplicate extracted ship JSON artifacts."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
import re
from typing import Dict, Iterable, List, Sequence, Tuple

from common.files import iter_json_files

MSG_SUFFIX_RE = re.compile(r"__msg\d+(?=(\.ship)?\.json$)")


@dataclass
class SourceFile:
    """Metadata about a single extracted JSON source file."""

    path: Path
    relpath: str
    content_hash: str
    stripped_name: str
    had_msg_suffix: bool


@dataclass
class CanonicalGroup:
    """A deduplicated canonical content group."""

    content_hash: str
    members: List[SourceFile]
    canonical_name: str
    canonical_name_source: str
    representative_path: Path


def canonicalize_json_text(text: str) -> Tuple[str, str]:
    """Normalize JSON text and return `(normalized_text, sha256_hash)`."""

    data = json.loads(text)
    normalized_text = json.dumps(
        data,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    content_hash = hashlib.sha256(normalized_text.encode("utf-8")).hexdigest()
    return normalized_text, content_hash


def strip_msg_suffix(name: str) -> Tuple[str, bool]:
    """Strip legacy Discord message suffixes from extracted JSON filenames."""

    stripped = MSG_SUFFIX_RE.sub("", name)
    return stripped, stripped != name


def choose_preferred_member(members: List[SourceFile]) -> Tuple[SourceFile, str]:
    """Choose the preferred representative file for a content group."""

    members_without_msg_suffix = [member for member in members if not member.had_msg_suffix]
    if members_without_msg_suffix:
        return (
            sorted(
                members_without_msg_suffix,
                key=lambda member: (
                    member.stripped_name.lower(),
                    member.stripped_name,
                    member.relpath,
                ),
            )[0],
            "existing_non_msg",
        )
    return (
        sorted(
            members,
            key=lambda member: (
                member.stripped_name.lower(),
                member.stripped_name,
                member.relpath,
            ),
        )[0],
        "stripped_msg_suffix",
    )


def resolve_collisions(
    groups: List[Tuple[str, List[SourceFile]]],
) -> Tuple[List[CanonicalGroup], List[dict]]:
    """Resolve canonical filename collisions across distinct content groups."""

    desired_name_to_groups: Dict[str, List[Tuple[str, List[SourceFile], str, Path]]] = defaultdict(list)

    for content_hash, members in groups:
        chosen_member, source_label = choose_preferred_member(members)
        representative_member = sorted(members, key=lambda member: member.relpath)[0]
        desired_name_to_groups[chosen_member.stripped_name].append(
            (content_hash, members, source_label, representative_member.path)
        )

    resolved_groups: List[CanonicalGroup] = []
    collision_records: List[dict] = []

    for desired_name in sorted(desired_name_to_groups, key=lambda name: (name.lower(), name)):
        entries = desired_name_to_groups[desired_name]
        suffix = "".join(Path(desired_name).suffixes)
        stem = desired_name[: -len(suffix)] if suffix else desired_name

        if len(entries) == 1:
            content_hash, members, source_label, representative_path = entries[0]
            resolved_groups.append(
                CanonicalGroup(
                    content_hash=content_hash,
                    members=members,
                    canonical_name=desired_name,
                    canonical_name_source=source_label,
                    representative_path=representative_path,
                )
            )
            continue

        ordered_entries = sorted(entries, key=lambda entry: entry[0])
        collision_record = {
            "desired_name": desired_name,
            "groups": [],
            "resolution_rule": (
                "Keep the lexicographically smallest SHA-256 content hash on the "
                "unsuffixed canonical name; append '__dedup-' + first 12 hash chars "
                "to the other colliding canonical filenames."
            ),
        }
        for index, (content_hash, members, source_label, representative_path) in enumerate(ordered_entries):
            final_name = desired_name if index == 0 else f"{stem}__dedup-{content_hash[:12]}{suffix}"
            resolved_groups.append(
                CanonicalGroup(
                    content_hash=content_hash,
                    members=members,
                    canonical_name=final_name,
                    canonical_name_source=source_label,
                    representative_path=representative_path,
                )
            )
            collision_record["groups"].append(
                {
                    "content_hash": content_hash,
                    "member_count": len(members),
                    "final_name": final_name,
                    "canonical_name_source": source_label,
                    "sample_sources": [
                        member.relpath
                        for member in sorted(members, key=lambda member: member.relpath)[:5]
                    ],
                }
            )
        collision_records.append(collision_record)

    return (
        sorted(
            resolved_groups,
            key=lambda group: (group.canonical_name.lower(), group.canonical_name),
        ),
        collision_records,
    )


def run_canonicalize(
    input_dir: str | Path = "extracted_ship_data",
    output_dir: str | Path = "extracted_ship_data_canonical",
    report_json: str | Path = "out/ship_canonicalization_report.json",
    report_md: str | Path = "SHIP_CANONICALIZATION_REPORT.md",
) -> dict:
    """Canonicalize and deduplicate extracted JSON files.

    Args:
        input_dir: Directory containing extracted JSON files
        output_dir: Directory where canonical JSON outputs will be written
        report_json: Machine-readable report destination
        report_md: Human-readable report destination

    Returns:
        The machine-readable manifest payload
    """

    input_path = Path(input_dir)
    output_path = Path(output_dir)
    report_json_path = Path(report_json)
    report_md_path = Path(report_md)

    files = list(iter_json_files(input_path))
    sources: List[SourceFile] = []
    parse_failures: List[dict] = []

    for index, path in enumerate(files, start=1):
        relpath = str(path.relative_to(input_path))
        try:
            text = path.read_text(encoding="utf-8")
            _, content_hash = canonicalize_json_text(text)
        except Exception as exc:  # pragma: no cover
            parse_failures.append({"file": relpath, "error": repr(exc)})
            continue

        stripped_name, had_msg_suffix = strip_msg_suffix(path.name)
        sources.append(
            SourceFile(
                path=path,
                relpath=relpath,
                content_hash=content_hash,
                stripped_name=stripped_name,
                had_msg_suffix=had_msg_suffix,
            )
        )

        if index % 1000 == 0:
            print(f"Scanned {index}/{len(files)} files...", flush=True)

    grouped_sources: Dict[str, List[SourceFile]] = defaultdict(list)
    for source in sources:
        grouped_sources[source.content_hash].append(source)

    resolved_groups, collisions = resolve_collisions(
        sorted(grouped_sources.items(), key=lambda item: item[0])
    )

    output_path.mkdir(parents=True, exist_ok=True)
    report_json_path.parent.mkdir(parents=True, exist_ok=True)

    manifest = {
        "input_dir": str(input_path),
        "output_dir": str(output_path),
        "total_input_json_files": len(files),
        "parsed_input_json_files": len(sources),
        "parse_failures": parse_failures,
        "unique_content_groups": len(resolved_groups),
        "duplicates_merged": len(sources) - len(resolved_groups),
        "canonical_names_from_stripping_msg_suffix": sum(
            1 for group in resolved_groups if group.canonical_name_source == "stripped_msg_suffix"
        ),
        "existing_non_msg_canonical_names": sum(
            1 for group in resolved_groups if group.canonical_name_source == "existing_non_msg"
        ),
        "filename_collision_count": len(collisions),
        "filename_collisions": collisions,
        "canonical_files": [],
        "duplicate_group_size_histogram": dict(
            sorted(Counter(len(group.members) for group in resolved_groups).items())
        ),
    }

    for index, group in enumerate(resolved_groups, start=1):
        normalized_text, recomputed_hash = canonicalize_json_text(
            group.representative_path.read_text(encoding="utf-8")
        )
        if recomputed_hash != group.content_hash:
            raise RuntimeError(
                "Representative file hash mismatch for "
                f"{group.representative_path}: expected {group.content_hash}, got {recomputed_hash}"
            )

        output_file_path = output_path / group.canonical_name
        output_file_path.write_text(normalized_text + "\n", encoding="utf-8")
        manifest["canonical_files"].append(
            {
                "canonical_name": group.canonical_name,
                "content_hash": group.content_hash,
                "canonical_name_source": group.canonical_name_source,
                "member_count": len(group.members),
                "representative_source_file": str(group.representative_path.relative_to(input_path)),
                "source_files": [
                    member.relpath for member in sorted(group.members, key=lambda member: member.relpath)
                ],
            }
        )

        if index % 1000 == 0:
            print(f"Wrote {index}/{len(resolved_groups)} canonical files...", flush=True)

    report_json_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")

    report_lines = [
        "# Ship JSON Canonicalization Report",
        "",
        f"- Input directory: `{input_path}`",
        f"- Canonical output directory: `{output_path}`",
        f"- Total input JSON files: **{manifest['total_input_json_files']}**",
        f"- Parsed input JSON files: **{manifest['parsed_input_json_files']}**",
        f"- Unique-content canonical JSON files: **{manifest['unique_content_groups']}**",
        f"- Duplicates merged: **{manifest['duplicates_merged']}**",
        (
            "- Canonical names that came from stripping `__msg<digits>`: "
            f"**{manifest['canonical_names_from_stripping_msg_suffix']}**"
        ),
        (
            "- Canonical names that already existed without `__msg`: "
            f"**{manifest['existing_non_msg_canonical_names']}**"
        ),
        f"- Filename collisions between different content groups: **{manifest['filename_collision_count']}**",
        "",
        "## Naming / dedupe rules",
        "",
        "1. Parse every `*.json` under the source corpus.",
        (
            "2. Canonicalize each parsed JSON as a minified object with stable "
            "recursive key ordering via `json.dumps(..., sort_keys=True, separators=(\",\", \":\"))`."
        ),
        "3. Hash the canonicalized JSON bytes with SHA-256 and dedupe by that hash, not by filename.",
        (
            "4. Keep only metadata in memory during the scan; re-read one representative "
            "source file per content group when writing outputs, so the full corpus does not "
            "need to stay resident in RAM."
        ),
        "5. Prefer a canonical filename that already exists without `__msg<digits>` when available.",
        "6. Otherwise, strip the `__msg<digits>` suffix from a representative filename.",
        (
            "7. If different content groups want the same canonical filename, keep the "
            "unsuffixed name for the lexicographically smallest content hash and append "
            "`__dedup-<12 hex>` to the rest."
        ),
        "",
        "## Duplicate group size histogram",
        "",
    ]

    for size, count in manifest["duplicate_group_size_histogram"].items():
        report_lines.append(f"- {count} group(s) with {size} file(s)")

    if parse_failures:
        report_lines.extend(["", "## Parse failures", ""])
        for item in parse_failures[:50]:
            report_lines.append(f"- `{item['file']}`: `{item['error']}`")
        if len(parse_failures) > 50:
            report_lines.append(f"- ... and {len(parse_failures) - 50} more")

    report_lines.extend(["", "## Filename collisions", ""])
    if not collisions:
        report_lines.append("- None")
    else:
        for collision in collisions:
            report_lines.append(
                f"- Desired canonical name `{collision['desired_name']}` had {len(collision['groups'])} distinct content groups."
            )
            report_lines.append(f"  - Resolution: {collision['resolution_rule']}")
            for group in collision["groups"]:
                report_lines.append(
                    "  - "
                    f"`{group['final_name']}` <- hash `{group['content_hash'][:12]}` "
                    f"from {group['member_count']} source file(s); "
                    f"naming source: `{group['canonical_name_source']}`"
                )

    report_lines.extend(["", "## Machine-readable detail", "", f"- Full manifest: `{report_json_path}`"])
    report_md_path.write_text("\n".join(report_lines) + "\n", encoding="utf-8")
    return manifest


def build_parser() -> argparse.ArgumentParser:
    """Build the CLI parser for canonicalization."""

    parser = argparse.ArgumentParser(
        description="Canonicalize and dedupe extracted ship JSON files by content."
    )
    parser.add_argument("--input-dir", default="extracted_ship_data")
    parser.add_argument("--output-dir", default="extracted_ship_data_canonical")
    parser.add_argument("--report-json", default="out/ship_canonicalization_report.json")
    parser.add_argument("--report-md", default="SHIP_CANONICALIZATION_REPORT.md")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run the canonicalization CLI."""

    parser = build_parser()
    args = parser.parse_args(argv)
    manifest = run_canonicalize(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        report_json=args.report_json,
        report_md=args.report_md,
    )
    summary = {
        key: manifest[key]
        for key in (
            "total_input_json_files",
            "parsed_input_json_files",
            "unique_content_groups",
            "duplicates_merged",
            "canonical_names_from_stripping_msg_suffix",
            "existing_non_msg_canonical_names",
            "filename_collision_count",
        )
    }
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
