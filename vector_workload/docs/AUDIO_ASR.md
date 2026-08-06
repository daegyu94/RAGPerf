# Audio ASR Vector Workload

이 workflow는 preparation server에서 audio를 ASR transcript로 변환하고 text embedding을 미리
생성한 뒤, target Milvus server에서 offline embedding artifact의 insert, DISKANN index build와
search/insert mixed schedule을 replay한다. Target server에는 audio decoder, ASR model 또는 GPU가
필요하지 않다.

Raw waveform embedding이나 acoustic similarity를 측정하는 workflow는 아니다. ASR과 text
embedding 시간은 artifact preparation에 속하며 Milvus replay latency에는 포함되지 않는다.

## Requirements

Repository root의 project-local environment를 사용한다.

```bash
source .venv/bin/activate
uv pip install -r vector_workload/requirements.txt
ffmpeg -version
```

ASR model과 text embedding model은 첫 실행 시 Hugging Face Hub에서 다운로드될 수 있다. 동일한
artifact를 재현하려면 두 model의 revision, ASR language, chunking, normalization과 dtype을
고정한다.

Query input은 기존 text query JSONL 형식이다.

```json
{"id":"query-001","text":"What was discussed in the recording?","delay_ms":0}
```

## Prepare ASR Transcripts

`audio-asr`는 audio directory를 재귀적으로 탐색하고 정렬된 순서로 transcript corpus를 만든다.
기본 지원 확장자는 WAV, FLAC, MP3, M4A, OGG와 Opus다.

```bash
python vector_workload/prepare_workloads.py audio-asr \
  --audio-dir /MNTPNT/ragperf/audio/files \
  --query-file /MNTPNT/ragperf/audio/queries.jsonl \
  --output-dir /MNTPNT/ragperf/audio/input \
  --model openai/whisper-small \
  --revision ASR_MODEL_REVISION \
  --device cuda:0 \
  --dtype float16 \
  --language en \
  --batch-size 8 \
  --chunk-length-seconds 30
```

CPU smoke에서는 `--model openai/whisper-tiny.en --device cpu --dtype float32`를 사용할 수 있다.
`.en` English-only model에서 `--language en`을 지정하면 preparer가 불필요한 model override를
자동으로 생략한다. `--audio-extensions wav flac`로 입력 형식을 제한하고
`--max-audio-files`로 작은 subset을 선택할 수 있다.

생성된 corpus row는 transcript를 `text`에 저장한다. `metadata`에는 다음 provenance가 포함된다.

- `modality: audio`, `representation: asr_transcript`
- 원본 audio의 상대 경로와 SHA-256 checksum
- ASR model, revision과 language

Raw audio는 portable artifact에 복사하지 않는다.

## Export Offline Embeddings

Transcript와 text query를 같은 text embedding model로 변환한다. Scheduled insert를 포함하려면
`--initial-corpus-ratio`를 `1`보다 작게 설정한다.

```bash
python vector_workload/export_vectors.py export \
  --corpus-file /MNTPNT/ragperf/audio/input/corpus.jsonl \
  --query-file /MNTPNT/ragperf/audio/input/queries.jsonl \
  --output-dir /MNTPNT/ragperf/audio/artifact \
  --model BAAI/bge-m3 \
  --revision TEXT_EMBEDDING_MODEL_REVISION \
  --device cuda:0 \
  --chunk-size 512 \
  --chunk-overlap 32 \
  --initial-corpus-ratio 0.8 \
  --searches-per-insert 10 \
  --insert-event-size 100

python vector_workload/export_vectors.py verify \
  --artifact-dir /MNTPNT/ragperf/audio/artifact
```

Audio ASR artifact는 표준 `single_vector` layout을 사용한다. `corpus` shard에는 initial row,
`inserts` shard에는 replay 중 추가할 pre-embedded row, `schedule` shard에는 search/insert 순서가
저장된다.

## Replay on Milvus

Artifact를 GPU가 없는 target server로 옮긴 뒤 [Replay Guide](REPLAY.md)의 표준 single-vector
절차로 실행한다. 이 artifact는 `COSINE` metric을 사용한다. Scheduled insert 뒤의 search에서 새
row를 확인해야 하면 기본값인 `Strong` consistency를 유지한다.
