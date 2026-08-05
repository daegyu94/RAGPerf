"""Shared checksum and schema validation for vector workload artifacts."""

from __future__ import annotations

import hashlib
from pathlib import Path

import pyarrow.parquet as pq
import yaml


SCHEMA_VERSION = 1


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def verify_artifact(artifact_dir: Path) -> None:
    artifact_dir = artifact_dir.resolve()
    manifest_path = artifact_dir / "workload-manifest.yaml"
    checksums_path = artifact_dir / "SHA256SUMS"
    if not manifest_path.is_file() or not checksums_path.is_file():
        raise FileNotFoundError("artifact requires workload-manifest.yaml and SHA256SUMS")

    with manifest_path.open("r", encoding="utf-8") as stream:
        manifest = yaml.safe_load(stream)
    if manifest.get("schema_version") != SCHEMA_VERSION:
        raise ValueError(f"unsupported schema version: {manifest.get('schema_version')}")

    expected_checksums: dict[str, str] = {}
    with checksums_path.open("r", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            parts = line.rstrip("\n").split("  ", maxsplit=1)
            if len(parts) != 2:
                raise ValueError(f"SHA256SUMS:{line_number}: malformed line")
            expected_checksums[parts[1]] = parts[0]

    for filename, expected in expected_checksums.items():
        path = artifact_dir / filename
        if not path.is_file():
            raise FileNotFoundError(f"missing artifact file: {filename}")
        actual = sha256_file(path)
        if actual != expected:
            raise ValueError(f"checksum mismatch for {filename}: {actual} != {expected}")

    manifest_artifacts = manifest.get("artifacts", [])
    if not manifest_artifacts:
        raise ValueError("manifest contains no artifacts")
    for artifact in manifest_artifacts:
        path = artifact_dir / artifact["path"]
        metadata = pq.read_metadata(path)
        if metadata.num_rows != artifact["rows"]:
            raise ValueError(
                f"row count mismatch for {path.name}: "
                f"{metadata.num_rows} != {artifact['rows']}"
            )
        if sha256_file(path) != artifact["sha256"]:
            raise ValueError(f"manifest checksum mismatch for {path.name}")
    print(f"Verified {len(manifest_artifacts)} Parquet shard(s) in {artifact_dir}")
