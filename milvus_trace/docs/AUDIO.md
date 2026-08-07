# Audio RAG 기록/재생

RAGPerf의 Audio RAG는 Hugging Face `datasets`의 오디오 샘플을 읽고,
Whisper 계열 ASR로 transcript를 만든 뒤 문장 임베딩을 생성합니다. Milvus
호출은 다른 workload와 동일한 `milvus_trace` recorder가 기록합니다.

## 빠른 시작

예시 설정은 [`config/milvus_audio.yaml`](../../config/milvus_audio.yaml)입니다.
`rag.audio.device`에는 `cpu` 또는 기록 환경에서 사용할 수 있는 `cuda:N`을
지정합니다. 먼저 기능을 확인할 때는 `cpu`와 작은 `sample_count`를 사용하는 것이
편리합니다.

Audio record에는 `torchcodec`가 필요합니다. `config/milvus_audio_smoke.yaml`은
`hf-internal-testing/librispeech_asr_dummy`의 `validation` split을 streaming으로 읽어
실제 오디오 8개만 처리합니다. 전체 workload는 `config/milvus_audio.yaml`의
`openslr/librispeech_asr`를 사용합니다. `torchcodec`가 사용하는 FFmpeg shared library도
설치되어 있어야 하며, `ffmpeg -version`으로 확인할 수 있습니다.

처음 실행하기 전에 [record host 설치](../README.md#record-host)를 완료합니다. Config의
`sys.vector_db.collection_name`과 `trace.output_dir`은 이전 run에서 사용하지 않은 값을
선택합니다.

```bash
export MNTPNT=/path/to/ragperf-data
export MILVUS_URI=http://milvus.example:19530
export RAG_DEVICE=cpu
export MSYS_CONFIG=config/monitor/example_config.yaml
export PYTHONPATH="$PWD:$PWD/src${PYTHONPATH:+:$PYTHONPATH}"
python src/run_new.py \
  --config config/milvus_audio.yaml \
  --msys-config "$MSYS_CONFIG"
```

처음 확인할 때는 config를 다음처럼 줄이면 정확히 8개 corpus sample과 최대 8개 query를
처리합니다.

```yaml
rag:
  audio:
    device: cpu
    sample_count: 8
  retrieval:
    question_num: 8
```

데이터셋 크기는 `rag.audio.sample_count`(정확한 샘플 수) 또는
`sample_count: null`일 때의 `bench.preprocessing.dataset_ratio`(유한 길이
dataset 비율)로 조절합니다. `rag.audio.streaming: true`에서는 전체 길이를 알 수
없으므로 `sample_count`를 반드시 지정해야 합니다. loader는 iterator와 bounded
batch만 사용하며 전체 오디오를 메모리에 저장하지 않습니다.

## 기록과 재생의 경계

기록 단계에서만 ASR와 embedding 모델이 필요합니다. insert에는 Milvus에 실제로
전달한 embedding, ASR transcript, JSON 호환 metadata가 저장됩니다. query 원문과
원본 오디오 바이트는 개인정보와 artifact 크기 문제 때문에 저장하지 않습니다.

replay 단계에서는 Whisper, `torch`, CUDA가 필요하지 않습니다. search event에
저장된 typed vector를 그대로 Milvus에 제출하므로 Audio RAG의 ASR 지연은 기록된
첫 search의 arrival timestamp와 이후 event 간격에 반영됩니다. ASR 계산이나 CPU/GPU
사용량 자체를 replay host에서 재현하는 것은 아닙니다.

## 지원 범위

- 지원 dataset: Hugging Face `load_dataset`로 읽을 수 있는 ASR dataset
- 기본값: `openslr/librispeech_asr`, `clean`, `train.100`
- 지원 작업: corpus insert, index 생성, audio query의 Milvus search
- generation/reranking: Audio RAG 경로에서는 선택적 후처리로 남겨 두며,
  trace artifact는 Milvus 요청까지만 재생합니다.

모델 다운로드나 오디오 decoder가 없는 환경에서는 recorder/replayer 단위 테스트를
먼저 실행하십시오. CPU-only replay image는 오디오 모델을 설치하지 않습니다.

native/Docker replay 명령은 [workload별 예시](WORKLOADS.md)를 참조하십시오.
