#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

MSG_SUFFIX_RE = re.compile(r"__msg\d+(?=(\.ship)?\.json$)")


@dataclass
class SourceFile:
    path: Path
    relpath: str
    content_hash: str
    stripped_name: str
    had_msg_suffix: bool


@dataclass
class CanonicalGroup:
    content_hash: str
    members: List[SourceFile]
    canonical_name: str
    canonical_name_source: str
    representative_path: Path


def canonicalize_json_text(text: str) -> Tuple[str, str]:
    data = json.loads(text)
    normalized_text = json.dumps(data, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    content_hash = hashlib.sha256(normalized_text.encode("utf-8")).hexdigest()
    return normalized_text, content_hash


def strip_msg_suffix(name: str) -> Tuple[str, bool]:
    stripped = MSG_SUFFIX_RE.sub("", name)
    return stripped, stripped != name


def choose_preferred_member(members: List[SourceFile]) -> Tuple[SourceFile, str]:
    without_msg = [m for m in members if not m.had_msg_suffix]
    if without_msg:
        return sorted(without_msg, key=lambda m: (m.stripped_name.lower(), m.stripped_name, m.relpath))[0], "existing_non_msg"
    return sorted(members, key=lambda m: (m.stripped_name.lower(), m.stripped_name, m.relpath))[0], "stripped_msg_suffix"


def resolve_collisions(groups: List[Tuple[str, List[SourceFile]]]) -> Tuple[List[CanonicalGroup], List[dict]]:
    desired_to_groups: Dict[str, List[Tuple[str, List[SourceFile], str, Path]]] = defaultdict(list)

    for content_hash, members in groups:
        chosen_member, source = choose_preferred_member(members)
        representative = sorted(members, key=lambda m: m.relpath)[0]
        desired_to_groups[chosen_member.stripped_name].append((content_hash, members, source, representative.path))

    resolved: List[CanonicalGroup] = []
    collisions: List[dict] = []

    for desired_name in sorted(desired_to_groups, key=lambda s: (s.lower(), s)):
        entries = desired_to_groups[desired_name]
        suffix = "".join(Path(desired_name).suffixes)
        stem = desired_name[: -len(suffix)] if suffix else desired_name

        if len(entries) == 1:
            content_hash, members, source, representative_path = entries[0]
            resolved.append(CanonicalGroup(content_hash, members, desired_name, source, representative_path))
            continue

        ordered = sorted(entries, key=lambda item: item[0])
        collision_record = {
            "desired_name": desired_name,
            "groups": [],
            "resolution_rule": "Keep the lexicographically smallest SHA-256 content hash on the unsuffixed canonical name; append '__dedup-' + first 12 hash chars to the other colliding canonical filenames.",
        }
        for index, (content_hash, members, source, representative_path) in enumerate(ordered):
            final_name = desired_name if index == 0 else f"{stem}__dedup-{content_hash[:12]}{suffix}"
            resolved.append(CanonicalGroup(content_hash, members, final_name, source, representative_path))
            collision_record["groups"].append(
                {
                    "content_hash": content_hash,
                    "member_count": len(members),
                    "final_name": final_name,
                    "canonical_name_source": source,
                    "sample_sources": [m.relpath for m in sorted(members, key=lambda m: m.relpath)[:5]],
                }
            )
        collisions.append(collision_record)

    return sorted(resolved, key=lambda g: (g.canonical_name.lower(), g.canonical_name)), collisions


def iter_json_files(input_dir: Path) -> Iterable[Path]:
    return sorted(path for path in input_dir.rglob("*.json") if path.is_file())


def main() -> None:
    parser = argparse.ArgumentParser(description="Canonicalize and dedupe extracted ship JSON files by content.")
    parser.add_argument("--input-dir", default="extracted_ship_data")
    parser.add_argument("--output-dir", default="extracted_ship_data_canonical")
    parser.add_argument("--report-json", default="out/ship_canonicalization_report.json")
    parser.add_argument("--report-md", default="SHIP_CANONICALIZATION_REPORT.md")
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    report_json_path = Path(args.report_json)
    report_md_path = Path(args.report_md)

    files = list(iter_json_files(input_dir))
    sources: List[SourceFile] = []
    parse_failures: List[dict] = []

    for index, path in enumerate(files, start=1):
        relpath = str(path.relative_to(input_dir))
        try:
            text = path.read_text()
            _, content_hash = canonicalize_json_text(text)
        except Exception as exc:  # pragma: no cover - surfaced in report
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

    grouped: Dict[str, List[SourceFile]] = defaultdict(list)
    for source in sources:
        grouped[source.content_hash].append(source)

    groups = sorted(grouped.items(), key=lambda item: item[0])
    resolved_groups, collisions = resolve_collisions(groups)

    output_dir.mkdir(parents=True, exist_ok=True)
    report_json_path.parent.mkdir(parents=True, exist_ok=True)

    manifest = {
        "input_dir": str(input_dir),
        "output_dir": str(output_dir),
        "total_input_json_files": len(files),
        "parsed_input_json_files": len(sources),
        "parse_failures": parse_failures,
        "unique_content_groups": len(resolved_groups),
        "duplicates_merged": len(sources) - len(resolved_groups),
        "canonical_names_from_stripping_msg_suffix": sum(1 for g in resolved_groups if g.canonical_name_source == "stripped_msg_suffix"),
        "existing_non_msg_canonical_names": sum(1 for g in resolved_groups if g.canonical_name_source == "existing_non_msg"),
        "filename_collision_count": len(collisions),
        "filename_collisions": collisions,
        "canonical_files": [],
        "duplicate_group_size_histogram": dict(sorted(Counter(len(group.members) for group in resolved_groups).items())),
    }

    for index, group in enumerate(resolved_groups, start=1):
        normalized_text, recomputed_hash = canonicalize_json_text(group.representative_path.read_text())
        if recomputed_hash != group.content_hash:
            raise RuntimeError(
                f"Representative file hash mismatch for {group.representative_path}: expected {group.content_hash}, got {recomputed_hash}"
            )

        out_path = output_dir / group.canonical_name
        out_path.write_text(normalized_text + "\n")
        manifest["canonical_files"].append(
            {
                "canonical_name": group.canonical_name,
                "content_hash": group.content_hash,
                "canonical_name_source": group.canonical_name_source,
                "member_count": len(group.members),
                "representative_source_file": str(group.representative_path.relative_to(input_dir)),
                "source_files": [m.relpath for m in sorted(group.members, key=lambda m: m.relpath)],
            }
        )

        if index % 1000 == 0:
            print(f"Wrote {index}/{len(resolved_groups)} canonical files...", flush=True)

    report_json_path.write_text(json.dumps(manifest, indent=2) + "\n")

    lines = [
        "# Ship JSON Canonicalization Report",
        "",
        f"- Input directory: `{input_dir}`",
        f"- Canonical output directory: `{output_dir}`",
        f"- Total input JSON files: **{manifest['total_input_json_files']}**",
        f"- Parsed input JSON files: **{manifest['parsed_input_json_files']}**",
        f"- Unique-content canonical JSON files: **{manifest['unique_content_groups']}**",
        f"- Duplicates merged: **{manifest['duplicates_merged']}**",
        f"- Canonical names that came from stripping `__msg<digits>`: **{manifest['canonical_names_from_stripping_msg_suffix']}**",
        f"- Canonical names that already existed without `__msg`: **{manifest['existing_non_msg_canonical_names']}**",
        f"- Filename collisions between different content groups: **{manifest['filename_collision_count']}**",
        "",
        "## Naming / dedupe rules",
        "",
        "1. Parse every `*.json` under the source corpus.",
        "2. Canonicalize each parsed JSON as a minified object with stable recursive key ordering via `json.dumps(..., sort_keys=True, separators=(\",\", \":\"))`.",
        "3. Hash the canonicalized JSON bytes with SHA-256 and dedupe by that hash, not by filename.",
        "4. Keep only metadata in memory during the scan; re-read one representative source file per content group when writing outputs, so the full corpus does not need to stay resident in RAM.",
        "5. Prefer a canonical filename that already exists without `__msg<digits>` when available.",
        "6. Otherwise, strip the `__msg<digits>` suffix from a representative filename.",
        "7. If different content groups want the same canonical filename, keep the unsuffixed name for the lexicographically smallest content hash and append `__dedup-<12 hex>` to the rest.",
        "",
        "## Duplicate group size histogram",
        "",
    ]

    for size, count in manifest["duplicate_group_size_histogram"].items():
        lines.append(f"- {count} group(s) with {size} file(s)")

    if parse_failures:
        lines.extend(["", "## Parse failures", ""])
        for item in parse_failures[:50]:
            lines.append(f"- `{item['file']}`: `{item['error']}`")
        if len(parse_failures) > 50:
            lines.append(f"- ... and {len(parse_failures) - 50} more")

    lines.extend(["", "## Filename collisions", ""])
    if not collisions:
        lines.append("- None")
    else:
        for collision in collisions:
            lines.append(f"- Desired canonical name `{collision['desired_name']}` had {len(collision['groups'])} distinct content groups.")
            lines.append(f"  - Resolution: {collision['resolution_rule']}")
            for group in collision["groups"]:
                lines.append(
                    f"  - `{group['final_name']}` <- hash `{group['content_hash'][:12]}` from {group['member_count']} source file(s); naming source: `{group['canonical_name_source']}`"
                )

    lines.extend(["", "## Machine-readable detail", "", f"- Full manifest: `{report_json_path}`"])
    report_md_path.write_text("\n".join(lines) + "\n")

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


if __name__ == "__main__":
    main()
