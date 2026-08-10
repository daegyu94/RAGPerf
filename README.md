# RAGPerf: An End-to-End Benchmarking Framework for Retrieval-Augmented Generation Systems <!-- omit from toc -->

**RAGPerf** is an open-source framework designed to benchmark the end-to-end system performance of Retrieval-Augmented Generation (RAG) applications. Built with a fully modular architecture, it offers a user-friendly and highly customizable framework that allows precise measurement of throughput, latency, and scalability across different RAG configurations.


<!-- Repo Characteristics -->
![CMake](https://img.shields.io/badge/CMake-008fba.svg?style=flat&logo=cmake&logoColor=ffffff)
![C++](https://img.shields.io/badge/c++-00599c.svg?style=flat&logo=c%2B%2B&logoColor=ffffff)
![Python](https://img.shields.io/badge/python-3670a0?style=flat&logo=python&logoColor=ffe465)
![OS Linux](https://img.shields.io/badge/OS-Linux-fcc624?style=flat&logo=linux&logoColor=ffffff)
[![Code style: clang-format](https://img.shields.io/badge/C/C++_Code_Style-clang--format-2a3e50?style=flat&logo=llvm&logoColor=cccccc)](resource/clang_format/.clang-format)
[![Code style: black](https://img.shields.io/badge/Python_Code_Style-black-000000?style=flat&logo=black&logoColor=ffffff)](resource/black_format/.black-format)

## Key Features

**🚀 Holistic System-Centric Benchmarking**: RAGPerf moves beyond simple accuracy metrics to profile the performance of RAG systems. It measures end-to-end throughput (QPS), latency breakdowns, and hardware efficiency. This helps developers identify potential bottlenecks throughout the entire pipeline.

**🧩 Modular Architecture**: RAGPerf uses a modular design that abstracts different stages of the RAG pipeline (Embedding, Vector Database, Reranking, and Generation) behind uniform interfaces. Users can seamlessly switch components (e.g., switching underlyinig vector database from Milvus to LanceDB, or change underlying generative model from GPT to Qwen) without rewriting code. This enables detailed performance comparisons between different system settings.

**📊 Detailed Full-Stack Profiling**: RAGPerf integrates a lightweight profiler that runs as a background daemon. It captures fine-grained hardware metrics with minimal overhead, including GPU/CPU utilization, memory consumptions (host RAM & GPU VRAM), PCIe throughput, and disk I/O utilization. This allows detailed analysis of resource utilization between RAG components and help identify potential system bottlenecks.

**🔄 Simulating Real-World Scenarios**: RAGPerf is able to simulate the evolution of real-world knowledge bases by synthesizing updates with a custom and configurable workload generator. The workload generator supports insert, update, and delete requests at different frequency and patterns, allowing users to estimate how data freshness and system performance varies in real systems.

**🖼️ Multi-Modal Capabilities**: RAGPerf supports diverse data modalities beyond plain text. It provides specialized pipelines including Visual RAG (PDFs, Images) using OCR or ColPali visual embeddings, and Audio RAG using ASR models like Whisper. This enables benchmarking of complex, unstructured RAG pipelines.

**⏺️ Milvus Request Trace Replay**: RAGPerf can record the Milvus requests produced by Text, Image, and Audio RAG workloads, then replay the same request payloads and arrival timing against another Milvus deployment without loading the original models or datasets.

---

<!-- omit from toc -->
## Table of Contents

- [Installation](#installation)
  - [Create a Virtual Environment](#create-a-virtual-environment)
  - [Install Dependencies](#install-dependencies)
  - [Install Monitoring System](#install-monitoring-system)
    - [C++ 20 Compatible Compiler Installation](#c-20-compatible-compiler-installation)
    - [Build MSys Shared Library and Position the Output Product to `src/monitoring_sys`](#build-msys-shared-library-and-position-the-output-product-to-srcmonitoring_sys)
- [Running RAGPerf](#running-ragperf)
  - [Quick Start with Web UI](#quick-start-with-web-ui)
    - [Preparation](#preparation)
    - [Configuring the Benchmark](#configuring-the-benchmark)
    - [Running the Benchmark](#running-the-benchmark)
  - [Run with Command Line Interface](#run-with-command-line-interface)
    - [Preparation](#preparation-1)
    - [Running the Benchmark](#running-the-benchmark-1)
    - [Performing Analysis](#performing-analysis)
  - [Record and Replay Milvus Workloads](#record-and-replay-milvus-workloads)
- [Supported RAG Pipeline Modules](#supported-rag-pipeline-modules)
  - [Vector Databases](#vector-databases)
  - [Monitoring System](#monitoring-system)

## Installation

### Create a Virtual Environment

RAGPerf는 package 충돌을 피하기 위해 독립된 Python 환경을 사용하는 것을 권장합니다. 아래에는
Python 기본 `venv`와 `conda` 두 가지 방법을 설명하며, 별도 Conda 설치가 없다면 `.venv` 방법을 사용하면 됩니다.

**venv (권장: Python 기본 환경)**
프로젝트를 Python 기본 `venv`에 설치하려면 repository root에서 다음을 실행합니다.

```bash
cd /path/to/RAGPerf
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
```

이후 이 문서의 `python` 명령은 모두 활성화한 `.venv`를 사용합니다. `venv` 모듈이
없다는 오류가 나오면 운영체제 패키지 관리자로 Python venv 지원 패키지를 먼저 설치합니다.

**Conda (recommended)**
```bash
# Install Miniconda/Mambaforge from the official site if you don't have Conda
conda create -n RAGPerf python=3.10
conda activate RAGPerf
```


### 전체 venv 설치 스크립트

처음부터 전체 RAGPerf를 설치할 때는 다음 스크립트를 사용합니다.

```bash
./milvus_trace/scripts/setup_venv.sh
source .venv/bin/activate
```

스크립트는 다음 작업을 순서대로 수행합니다.

- repository root에 `.venv`를 만들거나 기존 환경을 재사용합니다.
- `pip-tools`로 `requirement.txt`를 생성하고 Python dependency를 설치합니다.
- CMake build directory를 구성하고 `libmsys_pymod`를 build합니다.

이 스크립트는 Docker, CUDA, system package, C++ compiler를 설치하지 않습니다. 실행 전에
Python 3.10 이상, CMake, C++ 20 compatible compiler를 준비합니다.

Monitoring build를 건너뛰거나 위치를 바꾸려면 다음 옵션을 사용합니다.

```bash
./milvus_trace/scripts/setup_venv.sh --skip-monitoring
./milvus_trace/scripts/setup_venv.sh --venv-dir /path/to/venv --build-dir /path/to/build
```

스크립트가 끝난 뒤에는 별도 shell activation이 필요하므로 `source .venv/bin/activate`를
실행합니다.
### Install Dependencies

아래 단계는 Conda 또는 `.venv`가 활성화된 상태에서 실행합니다. `pip-tools`로 의존성 목록을 생성한 뒤 같은 환경에 설치합니다.
이 단계의 CMake 설정에는 C/C++ compiler가 필요하므로, compiler가 없다면 아래 `Install Monitoring System` 절차를 먼저 진행합니다.

```bash
# install pip-compile for Python package dependency resolution
python -m pip install "pip==25.3" "pip-tools==7.5.2"

# generate the project requirements file at the repository root
cmake -S . -B build
cmake --build build --target generate_py3_requirements

# install the dependencies into the active environment
python -m pip install -r requirement.txt
```

### Install Monitoring System

<!-- REVIEW: Put installation instructions here instead of readme in monitoring system module -->
RAGPerf uses a custom, low-overhead monitoring daemon. Here is a stripped-down version of the installation procedure (please refer to [MonitoringSystem README](monitoring_sys/README.md) for detailed instructions and explanations).

#### C++ 20 Compatible Compiler Installation

`venv`는 Python package만 관리하므로 C++ compiler는 운영체제에 설치해야 합니다. 먼저
다음으로 compiler가 있는지 확인합니다.

```bash
# check for a C++ 20 compatible compiler
g++ --version
```

Ubuntu/Debian에서 compiler가 없다면 다음을 실행합니다.

```bash
sudo apt-get update
sudo apt-get install -y build-essential cmake
```

Conda 사용자는 다음처럼 환경 안에 compiler를 설치할 수도 있습니다.

```bash
conda install -c conda-forge gcc=12.1.0
```

#### Build MSys Shared Library and Position the Output to `src/monitoring_sys`

Repository root에서 다음을 실행합니다.

```bash
cmake --build build --target libmsys_pymod --parallel 2
```

Make sure you see the file `libmsys.cpython-310-x86_64-linux-gnu.so` (the exact name could depend on your python version and architecture), this is the *cpython* module for the monitoring system executable.

## Running RAGPerf

RAGPerf provides an Interactive Web UI for ease of use. And of course, you can use the Command Line (CLI) for automation.

### Quick Start with Web UI

#### Preparation

Set these once in your shell rc file (e.g., `~/.bashrc` or `~/.zshrc`) or export them in every new shell.

```bash
# Make local "src" importable
export PYTHONPATH="$REPO_ROOT/src${PYTHONPATH+:$PYTHONPATH}"

# Where to cache Hugging Face models (optional, adjust path as needed)
export HF_HOME="/mnt/data/hf_home"
```

Install `streamlit` and run the RAGPerf client.

```bash
# install streamlit
python -m pip install streamlit
# run RAGPerf
streamlit run ui_client.py
```

Open the UI with the reported url in your web browser, the default url is `http://localhost:8501`.

#### Configuring the Benchmark

To run the benchmark, we first need to set up the vector database (See [vectordb](#vectordb) for more details). Then, customize your own workload settings with all the available options on the webpage.

![config](./doc/figures/ragconfig.png)

#### Running the Benchmark

In the execute page, click the `START BENCHMARK` button to execute the workload already configured. You may also want to check if all the configs are set correctly, see [here](./config/README.md) for detailed explanation of different entries in the config file.

![config](./doc/figures/run.png)

### Run with Command Line Interface

#### Preparation

Set these environment variables once in your shell rc file (e.g., `~/.bashrc` or `~/.zshrc`) or export them in every new shell.

```bash
# Make local `src` module importable
# set variable REPO_ROOT to correct path to the repo
export PYTHONPATH="$REPO_ROOT/src$PYTHONPATH"

# Where to cache Hugging Face models (optional, adjust path as needed)
export HF_HOME="/mnt/data/hf_home"
```

#### Running the Benchmark

To run the benchmark, you first need to set up the vector database as the retriever. See [vectordb](#vectordb) for a supported list and quick setup guide. Change the db_path to your local vector database storage path in config file.

```yaml
vector_db:
    db_path: /mnt/data/vectordb
```

First run the **preprocess/insert** phase to insert the dataset.

```bash
# 1) Build/insert into the vector store (LanceDB example)
python src/run_new.py \
  --config config/lance_insert.yaml \
  --msys-config config/monitor/example_config.yaml
```

After the insertion stage, proceed to the **query/evaluate** stage.

```bash
# 2) Retrieval and Query
python src/run_new.py \
  --config config/lance_query.yaml \
  --msys-config config/monitor/example_config.yaml
```

To customize your own workload setting, you may refer to the provided config file within `config` folder. The detailed parameters are listed [here](config/README.md).

#### Performing Analysis

You can check the output result within the `output` folder. To visualize the output results, run `python example/monitoring_sys_lib/test_parser.py`, the visualized figures will be located within the `output`.

### Record and Replay Milvus Workloads

Use the [Milvus trace guide](milvus_trace/README.md) to record the requests generated by a complete RAG run and replay them against a Milvus endpoint. The guide covers the one-command standalone server helper, an optional artifact-format check, the actual reduced-scale record→replay smoke test, Text/Image/Audio examples, native replay, Docker replay, artifact validation, and troubleshooting.

The replay tool contains a Milvus client, not a Milvus server. Run `./milvus_trace/scripts/milvus-standalone.sh start` before the first local test. The same endpoint can be used for source and target, with a different collection name for replay. Use separate deployments only when comparing environments.

## Supported RAG Pipeline Modules

### Vector Databases

RAGPerf supports many popular vector databases. To set up, check the detailed documentations at [VectorDB README](src/vectordb/README.md).

Want to add a new DB? Check our RAGPerf API at [VectorDB API](src/vectordb/README.md#adding-a-new-vector-database). This benchmark suit can automatically perform profiling and analysis on your vector database after implementing these APIs.

### Monitoring System

Examples of how to use the monitoring system are documented in `example/monitoring_sys_lib`. Detailed documentations at [MonitoringSystem README](monitoring_sys/README.md).

## Citation

Please consider citing us if you find RAGPerf useful in your research, the paper is available at arXiv: 

```bibtex
@misc{ragperf2026,
      title={RAGPerf: An End-to-End Benchmarking Framework for Retrieval-Augmented Generation Systems}, 
      author={Shaobo Li and Yirui Zhou and Yuan Xu and Kevin Chen and Daniel Waddington and Swaminathan Sundararaman and Hubertus Franke and Jian Huang},
      year={2026},
      eprint={2603.10765},
      archivePrefix={arXiv},
      primaryClass={cs.PF},
      url={https://arxiv.org/abs/2603.10765}, 
}
```
