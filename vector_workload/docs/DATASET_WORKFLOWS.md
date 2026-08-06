# Dataset Workflows

이 문서는 dataset별 input preparation recipe만 다룬다. 공통 artifact 기록 규칙과
`initial-corpus-ratio`, search/insert schedule의 의미는 [Record Guide](RECORD.md)가 기준이다.
Milvus 실행과 result 해석은 [Replay Guide](REPLAY.md), Docker Milvus 기동은
[Milvus Docker Setup](DOCKER_MILVUS.md)을 참고한다.

모든 명령은 repository root의 project-local `.venv`에서 실행한다. 환경 준비는
[Record Guide](RECORD.md)의 [Environment](RECORD.md#environment)를 따른다.

## Wikipedia + Natural Questions

먼저 corpus와 query JSONL을 만든다.

```bash
python vector_workload/prepare_workloads.py wikipedia-nq \
  --output-dir /MNTPNT/ragperf/wikipedia-nq/input \
  --corpus-count 100000 \
  --query-count 2000
```

준비된 두 JSONL을 Text Record command에 전달한다. 아래 예시는 corpus의 80%를 initial load로
사용하고 나머지를 scheduled insert로 replay하는 mixed workload다. 이 option들의 일반적인
의미는 [Record Guide](RECORD.md#defined-search-and-insert-order)에 정의되어 있다.

```bash
python vector_workload/record_workload.py text \
  --corpus-file /MNTPNT/ragperf/wikipedia-nq/input/corpus.jsonl \
  --query-file /MNTPNT/ragperf/wikipedia-nq/input/queries.jsonl \
  --output-dir /MNTPNT/ragperf/wikipedia-nq/record \
  --initial-corpus-ratio 0.8 \
  --searches-per-insert 10 \
  --insert-event-size 100
```

`--corpus-count`는 원본 Wikipedia document 수이며 최종 vector row 수가 아니다. Text chunking
후의 실제 row 수와 1TB 수준의 scale 계산은 [Record Guide](RECORD.md#estimate-dataset-and-storage-size)의
pilot 절차를 따른다. `input`은 보관하고 `record`를 새 directory로 만들어 다른 model이나
schedule을 반복할 수 있다.

## arXiv PDF Text

PDF page text와 query JSONL을 만든다. Text Record command의 기본 chunking을 사용한다.

```bash
python vector_workload/prepare_workloads.py arxiv-pdf-text \
  --pdf-dir /MNTPNT/ragperf/arxiv/pdfs \
  --query-file /MNTPNT/ragperf/arxiv/queries.jsonl \
  --output-dir /MNTPNT/ragperf/arxiv-text/input \
  --max-pdfs 10000

python vector_workload/record_workload.py text \
  --corpus-file /MNTPNT/ragperf/arxiv-text/input/corpus.jsonl \
  --query-file /MNTPNT/ragperf/arxiv-text/input/queries.jsonl \
  --output-dir /MNTPNT/ragperf/arxiv-text/record
```

Custom chunk size나 model revision이 필요하면 [Record Guide](RECORD.md)의 Text Embedding
command와 `record_workload.py text --help`를 기준으로 지정한다.

## arXiv PDF Images + ColPali

PDF page image를 준비한 뒤 ColPali Record command로 multi-vector artifact를 만든다.

```bash
python vector_workload/prepare_workloads.py arxiv-pdf-image \
  --pdf-dir /MNTPNT/ragperf/arxiv/pdfs \
  --query-file /MNTPNT/ragperf/arxiv/queries.jsonl \
  --output-dir /MNTPNT/ragperf/arxiv-image/input \
  --max-pdfs 10000

python vector_workload/record_workload.py colpali \
  --pdf-dir /MNTPNT/ragperf/arxiv/pdfs \
  --query-file /MNTPNT/ragperf/arxiv/queries.jsonl \
  --output-dir /MNTPNT/ragperf/arxiv-image/record \
  --max-pdfs 10000
```

`record_workload.py colpali`가 PDF rendering을 포함하므로, 위 예시에서 별도의
`arxiv-pdf-image` preparation 단계는 input을 미리 확인하거나 재사용할 때만 필요하다. 일반적인
실행에서는 두 번째 command만 사용해도 된다. `--max-pdfs`는 PDF 파일 수의 상한이며 실제
corpus document 수는 PDF page 수다. 한 page가 여러 token vector로 평탄화되므로 artifact row
수는 page 수보다 더 많다. Page 수와 실제 multi-vector row 수를 이용한 scale 계산은
[Record Guide](RECORD.md#colpali-pdf-image)의 pilot 절차를 따른다. ColPali artifact는
document/token grouping을 보존하며 Replay Guide가 `IP` metric과 multi-vector replay를 자동
선택한다.

이미 render한 page image를 재사용하려면 high-level wrapper가 PDF를 다시 render하므로, 준비된
`corpus.jsonl`을 low-level exporter에 직접 전달한다.

```bash
python vector_workload/export_colpali.py \
  --corpus-file /MNTPNT/ragperf/arxiv-image/input/corpus.jsonl \
  --query-file /MNTPNT/ragperf/arxiv-image/input/queries.jsonl \
  --output-dir /MNTPNT/ragperf/arxiv-image/record/artifact \
  --model vidore/colpali-v1.2 \
  --device cuda:0 \
  --batch-size 1

python vector_workload/export_vectors.py verify \
  --artifact-dir /MNTPNT/ragperf/arxiv-image/record/artifact
```

## Audio ASR + Text Embedding

Audio file을 transcript corpus로 만들고 text embedding artifact로 기록한다. `ffmpeg`가 필요하며,
directory를 재귀적으로 탐색해 WAV, FLAC, MP3, M4A, OGG와 Opus file을 정렬된 순서로 처리한다.
Query는 일반 text query JSONL 형식이다.

```json
{"id":"query-001","text":"What was discussed in the recording?","delay_ms":0}
```

```bash
ffmpeg -version

python vector_workload/record_workload.py audio-asr \
  --audio-dir /MNTPNT/ragperf/audio/files \
  --query-file /MNTPNT/ragperf/audio/queries.jsonl \
  --output-dir /MNTPNT/ragperf/audio/record \
  --max-audio-files 100
```

기본 ASR model은 `openai/whisper-small`, text embedding model은 `BAAI/bge-m3`이며 둘 다
`cuda:0`을 사용한다. ASR와 embedding을 서로 다른 device에서 실행하려면 `--asr-device`와
`--embedding-device`를 각각 지정한다. CPU smoke나 작은 subset에는 다음 option을 추가한다.

```text
--asr-model openai/whisper-tiny.en
--asr-device cpu
--asr-dtype float32
--max-audio-files 10
```

Corpus metadata에는 원본 audio의 상대 경로와 SHA-256, ASR model/revision/language가 기록된다.
Raw audio 자체는 artifact에 복사되지 않는다. 동일한 artifact를 재현하려면 ASR와 text embedding
model revision, language, chunking, normalization과 dtype을 고정한다. 결과는 `audio/artifact`에
저장되며 [Replay Guide](REPLAY.md)의 Audio ASR 예시로 실행한다. 위 예시에서는
`audio/record/artifact`에 저장된다. `--max-audio-files`는 pilot
크기를 제한할 뿐이며, 실제 vector row 수는 transcript 길이와 text chunking에 따라 결정된다.
먼저 100~1,000개 audio를 record해 manifest의 `inputs.corpus.chunks`를 확인한 뒤 파일 수를
확대한다. Audio 파일 자체의 저장 공간과 artifact 저장 공간은 별도로 계산한다.

## Synthetic Workload

Embedding model 없이 VectorDB I/O 경로를 확인할 때 synthetic artifact를 직접 생성한다.
Synthetic workload는 실제 dataset의 embedding 품질을 평가하지 않는다.

```bash
RUN_DIR=/MNTPNT/ragperf/vector-workload-100k-001

python vector_workload/generate_synthetic.py \
  --output-dir "$RUN_DIR/artifact" \
  --corpus-count 100000 \
  --query-count 2000 \
  --dimension 384 \
  --dtype float32 \
  --rows-per-shard 25000 \
  --query-noise 0.01 \
  --seed 42
```

Synthetic은 `--corpus-count`가 곧 corpus vector row 수이므로 다른 workload보다 1TB scale을
직접 조절하기 쉽다. Dimension과 dtype에 따른 raw vector 계산 및 Milvus 여유 공간은
[Record Guide](RECORD.md#synthetic)의 표를 기준으로 한다. Synthetic은 embedding 품질이나
실제 document chunk 분포를 평가하지 않으며, 대규모 I/O와 index 경로를 검증할 때 사용한다.

생성된 artifact는 [Artifact Format](ARTIFACT_FORMAT.md)의 contract를 따르며,
[Replay Guide](REPLAY.md)의 공통 replay command에 `artifact` 경로를 전달한다.

## Scope

이 문서에는 `export_vectors.py`, `export_colpali.py`, `replay_milvus.py`의 상세 option이나
Milvus index/data path 설정을 복사하지 않는다. 해당 내용의 기준 문서는 각각
[Record Guide](RECORD.md), [Replay Guide](REPLAY.md), [Milvus Docker Setup](DOCKER_MILVUS.md)다.
