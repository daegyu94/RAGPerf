"""Milvus trace artifact validation and streaming readers."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Iterator

import pyarrow.parquet as pq
import yaml


FORMAT_NAME = "ragperf-milvus-trace"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_manifest(artifact_dir: Path | str) -> dict:
    path = Path(artifact_dir) / "workload-manifest.yaml"
    if not path.is_file():
        raise FileNotFoundError(f"missing artifact manifest: {path}")
    with path.open("r", encoding="utf-8") as stream:
        manifest = yaml.safe_load(stream)
    if not isinstance(manifest, dict):
        raise ValueError("workload-manifest.yaml must contain a mapping")
    return manifest


def verify_artifact(artifact_dir: Path | str) -> dict:
    artifact_dir = Path(artifact_dir).resolve()
    manifest = load_manifest(artifact_dir)
    format_name = manifest.get("format")
    if format_name != FORMAT_NAME:
        raise ValueError(f"unsupported artifact format: {format_name}")
    if manifest.get("incomplete", False):
        raise ValueError("artifact is incomplete and cannot be replayed")

    sums_path = artifact_dir / "SHA256SUMS"
    if not sums_path.is_file():
        raise FileNotFoundError(f"missing artifact checksums: {sums_path}")
    expected = {}
    with sums_path.open("r", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            parts = line.rstrip("\n").split("  ", maxsplit=1)
            if len(parts) != 2 or Path(parts[1]).name != parts[1]:
                raise ValueError(f"SHA256SUMS:{line_number}: malformed line")
            expected[parts[1]] = parts[0]
    for filename, checksum in expected.items():
        path = artifact_dir / filename
        if not path.is_file():
            raise FileNotFoundError(f"missing artifact file: {filename}")
        if sha256_file(path) != checksum:
            raise ValueError(f"checksum mismatch for {filename}")

    for item in manifest.get("artifacts", []):
        path = artifact_dir / item["path"]
        metadata = pq.read_metadata(path)
        if metadata.num_rows != item["rows"]:
            raise ValueError(f"row count mismatch for {path.name}")
        if item.get("sha256") and sha256_file(path) != item["sha256"]:
            raise ValueError(f"manifest checksum mismatch for {path.name}")
    return manifest


def iter_parquet_rows(path: Path, batch_size: int = 1024) -> Iterator[dict]:
    parquet_file = pq.ParquetFile(path)
    for batch in parquet_file.iter_batches(batch_size=batch_size):
        yield from batch.to_pylist()


def iter_manifest_rows(
    artifact_dir: Path | str, manifest: dict, prefix: str, batch_size: int = 1024
) -> Iterator[dict]:
    artifact_dir = Path(artifact_dir)
    for item in manifest.get("artifacts", []):
        if item.get("kind") == prefix or item["path"].startswith(f"{prefix}-"):
            yield from iter_parquet_rows(artifact_dir / item["path"], batch_size)
