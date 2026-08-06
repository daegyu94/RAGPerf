#!/usr/bin/env python
"""Prepare canonical corpus/query inputs for portable RAGPerf workloads."""

from __future__ import annotations

import argparse
import inspect
import json
import shutil
import sys
from pathlib import Path
from typing import Any, Callable, Iterable

try:
    from artifact_utils import sha256_file
except ModuleNotFoundError:  # Supports `python -m vector_workload.prepare_workloads`.
    from vector_workload.artifact_utils import sha256_file


DEFAULT_ASR_MODEL = "openai/whisper-small"
DEFAULT_AUDIO_EXTENSIONS = (".flac", ".m4a", ".mp3", ".ogg", ".opus", ".wav")


def ensure_empty_output_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)
    if any(path.iterdir()):
        raise FileExistsError(f"output directory must be empty: {path}")


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> int:
    count = 0
    with path.open("w", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")
            count += 1
    if count == 0:
        raise ValueError(f"no records written to {path}")
    return count


def take_rows(rows: Iterable[dict[str, Any]], count: int) -> Iterable[dict[str, Any]]:
    iterator = iter(rows)
    try:
        for _ in range(count):
            try:
                yield next(iterator)
            except StopIteration:
                return
    finally:
        close = getattr(iterator, "close", None)
        if close is not None:
            close()


def wikipedia_rows(rows: Iterable[dict[str, Any]]) -> Iterable[dict[str, Any]]:
    for index, row in enumerate(rows):
        text = row.get("text")
        if not isinstance(text, str) or not text.strip():
            continue
        source_id = str(row.get("id", f"wikipedia-{index:08d}"))
        yield {
            "id": source_id,
            "text": text,
            "metadata": {
                "dataset": "wikimedia/wikipedia",
                "title": row.get("title"),
                "url": row.get("url"),
            },
        }


def natural_question_rows(rows: Iterable[dict[str, Any]]) -> Iterable[dict[str, Any]]:
    for index, row in enumerate(rows):
        query = row.get("query")
        if not isinstance(query, str) or not query.strip():
            continue
        yield {
            "id": str(row.get("id", f"natural-question-{index:08d}")),
            "text": query,
            "metadata": {
                "dataset": "sentence-transformers/natural-questions",
                "answer": row.get("answer"),
            },
            "delay_ms": 0,
        }


def prepare_wikipedia_nq(
    args: argparse.Namespace,
    load_dataset_fn: Callable[..., Iterable[dict[str, Any]]] | None = None,
) -> dict[str, Any]:
    if args.corpus_count <= 0 or args.query_count <= 0:
        raise ValueError("corpus-count and query-count must be positive")
    if load_dataset_fn is None:
        from datasets import load_dataset

        load_dataset_fn = load_dataset

    output_dir = args.output_dir.resolve()
    ensure_empty_output_dir(output_dir)
    wikipedia = load_dataset_fn(
        "wikimedia/wikipedia",
        "20231101.en",
        split="train",
        streaming=True,
    )
    natural_questions = load_dataset_fn(
        "sentence-transformers/natural-questions",
        split="train",
        streaming=True,
    )
    corpus_count = write_jsonl(
        output_dir / "corpus.jsonl",
        take_rows(wikipedia_rows(wikipedia), args.corpus_count),
    )
    query_count = write_jsonl(
        output_dir / "queries.jsonl",
        take_rows(natural_question_rows(natural_questions), args.query_count),
    )
    result = {
        "workload": "wikipedia-natural-questions",
        "output_dir": str(output_dir),
        "corpus_documents": corpus_count,
        "queries": query_count,
    }
    print(json.dumps(result, indent=2))
    return result


def pdf_text_rows(
    pdf_paths: Iterable[Path],
    reader_factory: Callable[[Path], Any] | None = None,
) -> Iterable[dict[str, Any]]:
    if reader_factory is None:
        from pypdf import PdfReader

        reader_factory = PdfReader
    for path in pdf_paths:
        reader = reader_factory(path)
        page_texts = [page.extract_text() or "" for page in reader.pages]
        text = "\n\n".join(part.strip() for part in page_texts if part.strip())
        if not text:
            continue
        yield {
            "id": path.stem,
            "text": text,
            "metadata": {
                "dataset": "common-pile/arxiv_papers",
                "source_file": path.name,
                "pages": len(reader.pages),
                "modality": "text",
            },
        }


def validate_query_file(path: Path) -> int:
    count = 0
    with path.open("r", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_number}: invalid JSON: {exc}") from exc
            if not isinstance(row, dict) or not isinstance(row.get("text"), str):
                raise ValueError(f"{path}:{line_number}: query requires string 'text'")
            count += 1
    if not count:
        raise ValueError(f"{path}: no queries found")
    return count


def prepare_arxiv_pdf_text(
    args: argparse.Namespace,
    reader_factory: Callable[[Path], Any] | None = None,
) -> dict[str, Any]:
    if args.max_pdfs is not None and args.max_pdfs <= 0:
        raise ValueError("max-pdfs must be positive")
    pdf_paths = sorted(args.pdf_dir.rglob("*.pdf"))
    if args.max_pdfs is not None:
        pdf_paths = pdf_paths[: args.max_pdfs]
    if not pdf_paths:
        raise ValueError(f"no PDF files found under {args.pdf_dir}")
    query_count = validate_query_file(args.query_file)

    output_dir = args.output_dir.resolve()
    ensure_empty_output_dir(output_dir)
    corpus_count = write_jsonl(
        output_dir / "corpus.jsonl", pdf_text_rows(pdf_paths, reader_factory)
    )
    shutil.copyfile(args.query_file, output_dir / "queries.jsonl")
    result = {
        "workload": "arxiv-pdf-text",
        "output_dir": str(output_dir),
        "pdf_documents": corpus_count,
        "queries": query_count,
    }
    print(json.dumps(result, indent=2))
    return result


def render_pdf_page_rows(
    pdf_paths: Iterable[Path],
    pages_dir: Path,
    converter: Callable[[Path], Iterable[Any]] | None = None,
) -> Iterable[dict[str, Any]]:
    if converter is None:
        import pypdfium2 as pdfium

        def converter(path: Path) -> Iterable[Any]:
            document = pdfium.PdfDocument(path)
            try:
                for page in document:
                    yield page.render(scale=2).to_pil()
            finally:
                document.close()

    pages_dir.mkdir(parents=True, exist_ok=True)
    for pdf_path in pdf_paths:
        for page_index, image in enumerate(converter(pdf_path), start=1):
            page_id = f"{pdf_path.stem}-page-{page_index:05d}"
            image_path = pages_dir / f"{page_id}.png"
            try:
                image.save(image_path, "PNG")
            finally:
                close = getattr(image, "close", None)
                if close is not None:
                    close()
            yield {
                "id": page_id,
                "image_path": str(image_path.resolve()),
                "metadata": {
                    "dataset": "common-pile/arxiv_papers",
                    "source_file": pdf_path.name,
                    "page": page_index,
                    "modality": "image",
                },
            }


def prepare_arxiv_pdf_image(
    args: argparse.Namespace,
    converter: Callable[[Path], Iterable[Any]] | None = None,
) -> dict[str, Any]:
    if args.max_pdfs is not None and args.max_pdfs <= 0:
        raise ValueError("max-pdfs must be positive")
    pdf_paths = sorted(args.pdf_dir.rglob("*.pdf"))
    if args.max_pdfs is not None:
        pdf_paths = pdf_paths[: args.max_pdfs]
    if not pdf_paths:
        raise ValueError(f"no PDF files found under {args.pdf_dir}")
    query_count = validate_query_file(args.query_file)

    output_dir = args.output_dir.resolve()
    ensure_empty_output_dir(output_dir)
    page_count = write_jsonl(
        output_dir / "corpus.jsonl",
        render_pdf_page_rows(pdf_paths, output_dir / "pages", converter),
    )
    shutil.copyfile(args.query_file, output_dir / "queries.jsonl")
    result = {
        "workload": "arxiv-pdf-image",
        "output_dir": str(output_dir),
        "pdf_documents": len(pdf_paths),
        "page_images": page_count,
        "queries": query_count,
    }
    print(json.dumps(result, indent=2))
    return result


def normalize_audio_extensions(extensions: Iterable[str]) -> set[str]:
    normalized = {
        extension.lower() if extension.startswith(".") else f".{extension.lower()}"
        for extension in extensions
        if extension.strip()
    }
    if not normalized:
        raise ValueError("at least one audio extension is required")
    return normalized


def find_audio_paths(
    audio_dir: Path,
    extensions: Iterable[str],
    max_audio_files: int | None,
) -> list[Path]:
    if max_audio_files is not None and max_audio_files <= 0:
        raise ValueError("max-audio-files must be positive")
    if not audio_dir.is_dir():
        raise ValueError(f"audio directory does not exist: {audio_dir}")
    allowed_extensions = normalize_audio_extensions(extensions)
    paths = sorted(
        path
        for path in audio_dir.rglob("*")
        if path.is_file() and path.suffix.lower() in allowed_extensions
    )
    if max_audio_files is not None:
        paths = paths[:max_audio_files]
    if not paths:
        extensions_text = ", ".join(sorted(allowed_extensions))
        raise ValueError(
            f"no audio files with extensions {extensions_text} found under {audio_dir}"
        )
    return paths


def resolve_asr_dtype(device: str, dtype: str) -> str:
    if dtype == "auto":
        return "float16" if device.startswith("cuda") else "float32"
    if dtype == "float16" and device == "cpu":
        raise ValueError("float16 ASR is not supported on CPU; use --dtype float32")
    return dtype


def load_asr_transcriber(
    model_name: str,
    revision: str | None,
    device: str,
    dtype: str,
) -> Callable[..., Any]:
    import torch
    from transformers import pipeline

    if device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError(
            f"CUDA device {device!r} requested, but CUDA is not available"
        )
    resolved_dtype = resolve_asr_dtype(device, dtype)
    pipeline_args: dict[str, Any] = {
        "task": "automatic-speech-recognition",
        "model": model_name,
        "device": device,
    }
    if revision is not None:
        pipeline_args["revision"] = revision
    dtype_value = getattr(torch, resolved_dtype)
    if "dtype" in inspect.signature(pipeline).parameters:
        pipeline_args["dtype"] = dtype_value
    else:
        pipeline_args["torch_dtype"] = dtype_value
    return pipeline(**pipeline_args)


def transcript_text(result: Any, path: Path) -> str:
    if isinstance(result, str):
        text = result
    elif isinstance(result, dict) and isinstance(result.get("text"), str):
        text = result["text"]
    else:
        raise ValueError(f"ASR returned no string transcript for {path}")
    return text.strip()


def audio_transcript_rows(
    audio_paths: Iterable[Path],
    audio_dir: Path,
    transcriber: Callable[..., Any],
    model_name: str,
    revision: str | None,
    language: str | None,
    chunk_length_seconds: float,
    batch_size: int,
    dataset_name: str,
) -> Iterable[dict[str, Any]]:
    transcribe_args: dict[str, Any] = {
        "chunk_length_s": chunk_length_seconds,
        "batch_size": batch_size,
    }
    if language is not None:
        normalized_language = language.lower()
        if model_name.lower().endswith(".en"):
            if normalized_language not in {"en", "english"}:
                raise ValueError(
                    f"English-only ASR model {model_name!r} cannot use language {language!r}"
                )
        else:
            transcribe_args["generate_kwargs"] = {"language": language}
    for index, path in enumerate(audio_paths):
        text = transcript_text(transcriber(str(path), **transcribe_args), path)
        if not text:
            continue
        relative_path = path.relative_to(audio_dir).as_posix()
        yield {
            "id": f"audio-{index:08d}",
            "text": text,
            "metadata": {
                "dataset": dataset_name,
                "source_file": relative_path,
                "source_sha256": sha256_file(path),
                "modality": "audio",
                "representation": "asr_transcript",
                "asr_model": model_name,
                "asr_revision": revision or "default",
                "asr_language": language or "auto",
            },
        }


def prepare_audio_asr(
    args: argparse.Namespace,
    transcriber: Callable[..., Any] | None = None,
) -> dict[str, Any]:
    if args.batch_size <= 0:
        raise ValueError("batch-size must be positive")
    if args.chunk_length_seconds <= 0:
        raise ValueError("chunk-length-seconds must be positive")
    audio_dir = args.audio_dir.resolve()
    audio_paths = find_audio_paths(
        audio_dir, args.audio_extensions, args.max_audio_files
    )
    query_count = validate_query_file(args.query_file)
    resolved_dtype = resolve_asr_dtype(args.device, args.dtype)
    if transcriber is None:
        transcriber = load_asr_transcriber(
            args.model, args.revision, args.device, resolved_dtype
        )

    output_dir = args.output_dir.resolve()
    ensure_empty_output_dir(output_dir)
    corpus_count = write_jsonl(
        output_dir / "corpus.jsonl",
        audio_transcript_rows(
            audio_paths,
            audio_dir,
            transcriber,
            args.model,
            args.revision,
            args.language,
            args.chunk_length_seconds,
            args.batch_size,
            args.dataset_name,
        ),
    )
    shutil.copyfile(args.query_file, output_dir / "queries.jsonl")
    result = {
        "workload": "audio-asr-text",
        "output_dir": str(output_dir),
        "audio_files_discovered": len(audio_paths),
        "audio_documents": corpus_count,
        "queries": query_count,
        "asr_model": args.model,
        "asr_revision": args.revision or "default",
        "asr_device": args.device,
        "asr_dtype": resolved_dtype,
    }
    print(json.dumps(result, indent=2))
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    wikipedia_parser = subparsers.add_parser(
        "wikipedia-nq", help="prepare Wikipedia corpus and Natural Questions queries"
    )
    wikipedia_parser.add_argument("--output-dir", type=Path, required=True)
    wikipedia_parser.add_argument("--corpus-count", type=int, required=True)
    wikipedia_parser.add_argument("--query-count", type=int, required=True)
    wikipedia_parser.set_defaults(func=prepare_wikipedia_nq)
    arxiv_text_parser = subparsers.add_parser(
        "arxiv-pdf-text", help="extract text from local arXiv PDF files"
    )
    arxiv_text_parser.add_argument("--pdf-dir", type=Path, required=True)
    arxiv_text_parser.add_argument("--query-file", type=Path, required=True)
    arxiv_text_parser.add_argument("--output-dir", type=Path, required=True)
    arxiv_text_parser.add_argument("--max-pdfs", type=int)
    arxiv_text_parser.set_defaults(func=prepare_arxiv_pdf_text)
    arxiv_image_parser = subparsers.add_parser(
        "arxiv-pdf-image", help="render local arXiv PDF files for ColPali"
    )
    arxiv_image_parser.add_argument("--pdf-dir", type=Path, required=True)
    arxiv_image_parser.add_argument("--query-file", type=Path, required=True)
    arxiv_image_parser.add_argument("--output-dir", type=Path, required=True)
    arxiv_image_parser.add_argument("--max-pdfs", type=int)
    arxiv_image_parser.set_defaults(func=prepare_arxiv_pdf_image)
    audio_parser = subparsers.add_parser(
        "audio-asr", help="transcribe local audio files into a text retrieval corpus"
    )
    audio_parser.add_argument("--audio-dir", type=Path, required=True)
    audio_parser.add_argument("--query-file", type=Path, required=True)
    audio_parser.add_argument("--output-dir", type=Path, required=True)
    audio_parser.add_argument("--model", default=DEFAULT_ASR_MODEL)
    audio_parser.add_argument("--revision")
    audio_parser.add_argument("--device", default="cuda:0")
    audio_parser.add_argument(
        "--dtype", choices=("auto", "float16", "float32"), default="auto"
    )
    audio_parser.add_argument("--language")
    audio_parser.add_argument("--batch-size", type=int, default=8)
    audio_parser.add_argument("--chunk-length-seconds", type=float, default=30)
    audio_parser.add_argument("--max-audio-files", type=int)
    audio_parser.add_argument(
        "--audio-extensions", nargs="+", default=DEFAULT_AUDIO_EXTENSIONS
    )
    audio_parser.add_argument("--dataset-name", default="local-audio-asr")
    audio_parser.set_defaults(func=prepare_audio_asr)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    try:
        args.func(args)
    except Exception as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
