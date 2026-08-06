#!/usr/bin/env python
"""Replay a recorded workload with artifact-aware Milvus defaults."""

from __future__ import annotations

import argparse
import subprocess
import sys
from collections.abc import Sequence
from pathlib import Path

import yaml


SCRIPT_DIR = Path(__file__).resolve().parent


def add_option(command: list[str], name: str, value: object | None) -> None:
    if value is not None:
        command.extend((name, str(value)))


def artifact_profile(artifact_dir: Path) -> tuple[str, str]:
    manifest_path = artifact_dir / "workload-manifest.yaml"
    with manifest_path.open("r", encoding="utf-8") as stream:
        manifest = yaml.safe_load(stream)
    if not isinstance(manifest, dict):
        raise ValueError(f"invalid manifest: {manifest_path}")
    embedding = manifest.get("embedding", {})
    if not isinstance(embedding, dict):
        raise ValueError(f"invalid embedding metadata: {manifest_path}")
    layout = embedding.get("vector_layout", "single_vector")
    if layout not in {"single_vector", "multi_vector"}:
        raise ValueError(f"unsupported vector layout: {layout}")
    return layout, "IP" if layout == "multi_vector" else "COSINE"


def build_commands(args: argparse.Namespace) -> tuple[list[str], list[str], str, Path]:
    artifact_dir = args.artifact_dir.resolve()
    layout, default_metric = artifact_profile(artifact_dir)
    metric = args.metric or default_metric
    result_file = (
        args.result_file.resolve()
        if args.result_file
        else artifact_dir.parent / "replay-result.json"
    )
    verify = [
        sys.executable,
        str(SCRIPT_DIR / "export_vectors.py"),
        "verify",
        "--artifact-dir",
        str(artifact_dir),
    ]
    replay = [
        sys.executable,
        str(SCRIPT_DIR / "replay_milvus.py"),
        "--artifact-dir",
        str(artifact_dir),
        "--uri",
        args.uri,
        "--collection",
        args.collection,
        "--result-file",
        str(result_file),
        "--metric",
        metric,
    ]
    add_option(replay, "--warmup-queries", args.warmup_queries)
    add_option(replay, "--concurrency", args.concurrency)
    add_option(replay, "--search-list", args.search_list)
    add_option(replay, "--top-k", args.top_k)
    add_option(replay, "--storage-path-note", args.storage_path_note)
    if layout == "multi_vector":
        add_option(replay, "--token-top-k", args.token_top_k)
    if args.respect_delay:
        replay.append("--respect-delay")
    return verify, replay, layout, result_file


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact-dir", type=Path, required=True)
    parser.add_argument("--collection", required=True)
    parser.add_argument("--uri", default="http://127.0.0.1:19530")
    parser.add_argument("--result-file", type=Path)
    parser.add_argument("--metric", choices=("COSINE", "L2", "IP"))
    parser.add_argument("--warmup-queries", type=int)
    parser.add_argument("--concurrency", type=int)
    parser.add_argument("--search-list", type=int)
    parser.add_argument("--top-k", type=int)
    parser.add_argument("--token-top-k", type=int, default=100)
    parser.add_argument("--storage-path-note")
    parser.add_argument("--respect-delay", action="store_true")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    try:
        verify, replay, layout, result_file = build_commands(args)
        print(
            f"Replay profile: {layout}, metric={replay[replay.index('--metric') + 1]}",
            flush=True,
        )
        subprocess.run(verify, check=True)
        subprocess.run(replay, check=True)
    except (OSError, subprocess.CalledProcessError, ValueError, yaml.YAMLError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    print(f"Replay result: {result_file}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
