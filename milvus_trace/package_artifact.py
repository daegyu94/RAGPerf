"""Create a deterministic tar.zst release asset from a trace artifact."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import tempfile
from pathlib import Path

from .artifact import verify_artifact


class PackagingError(RuntimeError):
    pass


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def package_artifact(
    artifact_dir: Path,
    output: Path | None = None,
    compression_level: int = 19,
    threads: int = 0,
) -> dict:
    artifact_dir = artifact_dir.resolve()
    manifest = verify_artifact(artifact_dir)
    tar = shutil.which("tar")
    zstd = shutil.which("zstd")
    if tar is None or zstd is None:
        raise PackagingError("tar and zstd executables are required")
    if not 1 <= compression_level <= 22:
        raise ValueError("compression level must be between 1 and 22")
    if threads < 0:
        raise ValueError("threads must be zero or positive")

    output = output or artifact_dir.with_name(f"{artifact_dir.name}.tar.zst")
    output = output.resolve()
    if output.parent == artifact_dir or artifact_dir in output.parents:
        raise ValueError("output archive must be outside artifact directory")
    if output.exists() or output.with_name(f"{output.name}.sha256").exists():
        raise FileExistsError(f"output already exists: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    partial_fd, partial_name = tempfile.mkstemp(
        prefix=f".{output.name}.", suffix=".part", dir=output.parent
    )
    os.close(partial_fd)
    partial = Path(partial_name)
    partial.unlink()
    tar_process = None
    zstd_process = None
    try:
        tar_process = subprocess.Popen(
            [
                tar,
                "--sort=name",
                "--mtime=UTC 1970-01-01",
                "--owner=0",
                "--group=0",
                "--numeric-owner",
                "-cf",
                "-",
                "-C",
                str(artifact_dir.parent),
                artifact_dir.name,
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        assert tar_process.stdout is not None
        zstd_process = subprocess.Popen(
            [zstd, f"-T{threads}", f"-{compression_level}", "-q", "-o", str(partial)],
            stdin=tar_process.stdout,
            stderr=subprocess.PIPE,
        )
        tar_process.stdout.close()
        zstd_stderr = zstd_process.communicate()[1].decode(errors="replace")
        tar_stderr = tar_process.communicate()[1].decode(errors="replace")
        if tar_process.returncode or zstd_process.returncode:
            raise PackagingError(
                "archive command failed: "
                f"tar={tar_process.returncode}, zstd={zstd_process.returncode}; "
                f"{tar_stderr}{zstd_stderr}"
            )
        os.replace(partial, output)
        subprocess.run([zstd, "-t", "-q", str(output)], check=True)
    except Exception:
        if tar_process is not None and tar_process.poll() is None:
            tar_process.kill()
        if zstd_process is not None and zstd_process.poll() is None:
            zstd_process.kill()
        partial.unlink(missing_ok=True)
        raise

    checksum = sha256_file(output)
    checksum_path = output.with_name(f"{output.name}.sha256")
    checksum_path.write_text(f"{checksum}  {output.name}\n", encoding="utf-8")
    return {
        "format": manifest["format"],
        "artifact_dir": str(artifact_dir),
        "archive": str(output),
        "archive_bytes": output.stat().st_size,
        "sha256": checksum,
        "sha256_file": str(checksum_path),
        "compression_level": compression_level,
        "threads": threads,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--compression-level", type=int, default=19)
    parser.add_argument("--threads", type=int, default=0)
    args = parser.parse_args(argv)
    print(
        json.dumps(
            package_artifact(
                args.artifact_dir,
                args.output,
                args.compression_level,
                args.threads,
            ),
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
