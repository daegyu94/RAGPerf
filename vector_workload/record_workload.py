#!/usr/bin/env python
"""Record text, Audio ASR, or ColPali inputs as a vector workload artifact."""

from __future__ import annotations

import argparse
import subprocess
import sys
from collections.abc import Callable, Sequence
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
CommandRunner = Callable[[Sequence[str]], None]


def run_command(command: Sequence[str]) -> None:
    subprocess.run(command, check=True)


def add_option(command: list[str], name: str, value: object | None) -> None:
    if value is not None:
        command.extend((name, str(value)))


def add_schedule_options(command: list[str], args: argparse.Namespace) -> None:
    command.extend(
        (
            "--initial-corpus-ratio",
            str(args.initial_corpus_ratio),
            "--searches-per-insert",
            str(args.searches_per_insert),
            "--insert-event-size",
            str(args.insert_event_size),
        )
    )


def prepare_run_dir(path: Path) -> Path:
    run_dir = path.resolve()
    run_dir.mkdir(parents=True, exist_ok=True)
    if any(run_dir.iterdir()):
        raise FileExistsError(f"output directory must be empty: {run_dir}")
    return run_dir


def verify_command(artifact_dir: Path) -> list[str]:
    return [
        sys.executable,
        str(SCRIPT_DIR / "export_vectors.py"),
        "verify",
        "--artifact-dir",
        str(artifact_dir),
    ]


def record_text(args: argparse.Namespace, runner: CommandRunner = run_command) -> Path:
    run_dir = prepare_run_dir(args.output_dir)
    artifact_dir = run_dir / "artifact"
    command = [
        sys.executable,
        str(SCRIPT_DIR / "export_vectors.py"),
        "export",
        "--corpus-file",
        str(args.corpus_file),
        "--query-file",
        str(args.query_file),
        "--output-dir",
        str(artifact_dir),
        "--device",
        args.device,
        "--batch-size",
        str(args.batch_size),
    ]
    if args.smoke:
        command.append("--smoke")
    add_option(command, "--model", args.model)
    add_option(command, "--revision", args.revision)
    add_schedule_options(command, args)
    runner(command)
    runner(verify_command(artifact_dir))
    return artifact_dir


def record_audio_asr(args: argparse.Namespace, runner: CommandRunner = run_command) -> Path:
    run_dir = prepare_run_dir(args.output_dir)
    input_dir = run_dir / "input"
    artifact_dir = run_dir / "artifact"
    prepare = [
        sys.executable,
        str(SCRIPT_DIR / "prepare_workloads.py"),
        "audio-asr",
        "--audio-dir",
        str(args.audio_dir),
        "--query-file",
        str(args.query_file),
        "--output-dir",
        str(input_dir),
        "--model",
        args.asr_model,
        "--device",
        args.asr_device,
        "--dtype",
        args.asr_dtype,
        "--batch-size",
        str(args.asr_batch_size),
    ]
    add_option(prepare, "--revision", args.asr_revision)
    add_option(prepare, "--language", args.language)
    add_option(prepare, "--max-audio-files", args.max_audio_files)
    add_option(prepare, "--chunk-length-seconds", args.chunk_length_seconds)
    add_option(prepare, "--dataset-name", args.dataset_name)
    if args.audio_extensions:
        prepare.extend(("--audio-extensions", *args.audio_extensions))
    runner(prepare)

    export = [
        sys.executable,
        str(SCRIPT_DIR / "export_vectors.py"),
        "export",
        "--corpus-file",
        str(input_dir / "corpus.jsonl"),
        "--query-file",
        str(input_dir / "queries.jsonl"),
        "--output-dir",
        str(artifact_dir),
        "--device",
        args.embedding_device,
        "--batch-size",
        str(args.embedding_batch_size),
    ]
    add_option(export, "--model", args.embedding_model)
    add_option(export, "--revision", args.embedding_revision)
    add_schedule_options(export, args)
    runner(export)
    runner(verify_command(artifact_dir))
    return artifact_dir


def record_colpali(args: argparse.Namespace, runner: CommandRunner = run_command) -> Path:
    run_dir = prepare_run_dir(args.output_dir)
    input_dir = run_dir / "input"
    artifact_dir = run_dir / "artifact"
    prepare = [
        sys.executable,
        str(SCRIPT_DIR / "prepare_workloads.py"),
        "arxiv-pdf-image",
        "--pdf-dir",
        str(args.pdf_dir),
        "--query-file",
        str(args.query_file),
        "--output-dir",
        str(input_dir),
    ]
    add_option(prepare, "--max-pdfs", args.max_pdfs)
    runner(prepare)

    export = [
        sys.executable,
        str(SCRIPT_DIR / "export_colpali.py"),
        "--corpus-file",
        str(input_dir / "corpus.jsonl"),
        "--query-file",
        str(input_dir / "queries.jsonl"),
        "--output-dir",
        str(artifact_dir),
        "--model",
        args.model,
        "--device",
        args.device,
        "--batch-size",
        str(args.batch_size),
    ]
    add_option(export, "--revision", args.revision)
    add_schedule_options(export, args)
    runner(export)
    runner(verify_command(artifact_dir))
    return artifact_dir


def add_schedule_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--initial-corpus-ratio", type=float, default=1.0)
    parser.add_argument("--searches-per-insert", type=int, default=1)
    parser.add_argument("--insert-event-size", type=int, default=1)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="workload", required=True)

    text = subparsers.add_parser("text", help="record text corpus and queries")
    text.add_argument("--corpus-file", type=Path, required=True)
    text.add_argument("--query-file", type=Path, required=True)
    text.add_argument("--output-dir", type=Path, required=True, help="new run directory")
    text.add_argument("--model")
    text.add_argument("--revision")
    text.add_argument("--device", default="cuda:0")
    text.add_argument("--batch-size", type=int, default=32)
    text.add_argument("--smoke", action="store_true")
    add_schedule_arguments(text)
    text.set_defaults(func=record_text)

    audio = subparsers.add_parser(
        "audio-asr", help="transcribe audio and record a text embedding artifact"
    )
    audio.add_argument("--audio-dir", type=Path, required=True)
    audio.add_argument("--query-file", type=Path, required=True)
    audio.add_argument("--output-dir", type=Path, required=True, help="new run directory")
    audio.add_argument("--asr-model", default="openai/whisper-small")
    audio.add_argument("--asr-revision")
    audio.add_argument("--asr-device", default="cuda:0")
    audio.add_argument("--asr-dtype", choices=("auto", "float16", "float32"), default="auto")
    audio.add_argument("--asr-batch-size", type=int, default=8)
    audio.add_argument("--language")
    audio.add_argument("--max-audio-files", type=int)
    audio.add_argument("--chunk-length-seconds", type=float)
    audio.add_argument("--audio-extensions", nargs="+")
    audio.add_argument("--dataset-name")
    audio.add_argument("--embedding-model")
    audio.add_argument("--embedding-revision")
    audio.add_argument("--embedding-device", default="cuda:0")
    audio.add_argument("--embedding-batch-size", type=int, default=32)
    add_schedule_arguments(audio)
    audio.set_defaults(func=record_audio_asr)

    colpali = subparsers.add_parser(
        "colpali", help="render PDF pages and record a ColPali artifact"
    )
    colpali.add_argument("--pdf-dir", type=Path, required=True)
    colpali.add_argument("--query-file", type=Path, required=True)
    colpali.add_argument("--output-dir", type=Path, required=True, help="new run directory")
    colpali.add_argument("--model", default="vidore/colpali-v1.2")
    colpali.add_argument("--revision")
    colpali.add_argument("--device", default="cuda:0")
    colpali.add_argument("--batch-size", type=int, default=1)
    colpali.add_argument("--max-pdfs", type=int)
    add_schedule_arguments(colpali)
    colpali.set_defaults(func=record_colpali)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    try:
        artifact_dir = args.func(args)
    except (OSError, subprocess.CalledProcessError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    print(f"Artifact ready: {artifact_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
