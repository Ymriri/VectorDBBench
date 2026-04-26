# VectorDBBench - AI Context Guide

> This file helps AI assistants understand and work with the VectorDBBench codebase effectively.

## Project Overview

**VectorDBBench** is an open-source benchmark tool for vector databases, maintained by Zilliz (creators of Milvus). It provides performance and cost-effectiveness comparisons across 40+ vector databases through GUI (Streamlit), CLI, and RESTful API interfaces.

- **Language**: Python 3.11+
- **License**: MIT
- **Repository**: https://github.com/zilliztech/VectorDBBench
- **Package**: `vectordb-bench` on PyPI

## Architecture

```
User Interface Layer
├── Streamlit Web UI (frontend/)     - Interactive benchmark runner
├── CLI Tool (cli/)                  - Command-line execution
└── RESTful API (restful/)           - HTTP API service

Core Business Logic (backend/)
├── BenchMarkRunner (interface.py)   - Task orchestration & async execution
├── CaseRunner (task_runner.py)      - Individual test case execution
├── Assembler (assembler.py)         - Task assembly from configs
├── Cases (cases.py)                 - Test case definitions
├── Dataset Manager (dataset.py)     - Dataset download & management
├── Runners (runner/)                - Execution engines (serial/concurrent/MP)
└── Results (models.py, metric.py)   - Metrics collection & persistence

Database Adapter Layer (backend/clients/)
├── api.py                           - VectorDB abstract base class
├── 40+ database implementations     - Milvus, PgVector, Elastic, Pinecone, etc.
└── __init__.py                      - DB enum & client registry

Data & Storage
├── S3 / AliyunOSS                   - Dataset source
├── Local parquet files              - Cached datasets
└── JSON results                     - Test output storage
```

## Directory Structure

```
VectorDBBench/
├── vectordb_bench/                  # Main package
│   ├── __init__.py                  # Global config (class config)
│   ├── __main__.py                  # GUI entry: launches Streamlit
│   ├── interface.py                 # Core API: BenchMarkRunner class
│   ├── models.py                    # Data models: TaskConfig, CaseResult, TestResult
│   ├── metric.py                    # Metrics: recall, ndcg, qps, latency
│   ├── base.py                      # BaseModel with common utilities
│   │
│   ├── backend/                     # Core benchmarking logic
│   │   ├── cases.py                 # CaseType enum, CapacityCase, PerformanceCase, StreamingPerformanceCase
│   │   ├── dataset.py               # Dataset enum, DatasetManager, DataSetIterator
│   │   ├── data_source.py           # S3/AliyunOSS dataset download
│   │   ├── filter.py                # Filter types: IntFilter, LabelFilter, NonFilter
│   │   ├── assembler.py             # Assembles TaskConfig into CaseRunners
│   │   ├── task_runner.py           # CaseRunner & TaskRunner execution logic
│   │   ├── result_collector.py      # Reads result JSON files
│   │   ├── utils.py                 # Utility functions
│   │   ├── clients/                 # Database client implementations
│   │   │   ├── api.py               # ABC: VectorDB, DBConfig, DBCaseConfig, MetricType, IndexType
│   │   │   ├── __init__.py          # DB enum, db2client mapping
│   │   │   ├── milvus/              # Milvus client
│   │   │   ├── zilliz_cloud/        # Zilliz Cloud client
│   │   │   ├── pgvector/            # PostgreSQL pgvector client
│   │   │   ├── elastic_cloud/       # Elastic Cloud client
│   │   │   ├── pinecone/            # Pinecone client
│   │   │   ├── qdrant_cloud/        # Qdrant client
│   │   │   ├── weaviate_cloud/      # Weaviate client
│   │   │   ├── redis/               # Redis client
│   │   │   ├── chroma/              # ChromaDB client
│   │   │   ├── mongodb/             # MongoDB client
│   │   │   ├── oceanbase/           # OceanBase client
│   │   │   └── ... (30+ more)
│   │   └── runner/                  # Execution engines
│   │       ├── serial_runner.py     # SerialInsertRunner, SerialSearchRunner
│   │       ├── concurrent_runner.py # ConcurrentInsertRunner
│   │       ├── mp_runner.py         # MultiProcessingSearchRunner
│   │       ├── read_write_runner.py # ReadWriteRunner (streaming)
│   │       ├── rate_runner.py       # Rate-controlled execution
│   │       └── executor.py          # Execution utilities
│   │
│   ├── cli/                         # Command-line interface
│   │   ├── vectordbbench.py         # CLI main entry
│   │   ├── cli.py                   # CommonTypedDict, run(), click decorators
│   │   └── batch_cli.py             # Batch execution from YAML config
│   │
│   ├── frontend/                    # Streamlit web UI
│   │   ├── vdbbench.py              # Streamlit entry point
│   │   ├── pages/                   # UI pages
│   │   │   ├── run_test.py          # Run benchmark page
│   │   │   ├── results.py           # Results display
│   │   │   ├── qps_recall.py        # QPS-Recall curves
│   │   │   ├── tables.py            # Table views
│   │   │   ├── custom.py            # Custom dataset page
│   │   │   └── ...
│   │   ├── components/              # Reusable UI components
│   │   └── config/                  # Frontend styling configs
│   │
│   ├── results/                     # Test result JSON storage
│   ├── config-files/                # Example YAML config files
│   └── custom/                      # Custom case configurations
│
├── tests/                           # Test suite
├── pyproject.toml                   # Project config, dependencies, entry points
├── install.py                       # Installation helper
├── Makefile                         # lint, format commands
└── Dockerfile                       # Container image
```

## Key Entry Points

| Entry | Command | File | Purpose |
|-------|---------|------|---------|
| GUI | `init_bench` | `__main__.py:run_streamlit()` | Launch Streamlit web UI |
| CLI | `vectordbbench [cmd] [opts]` | `cli/vectordbbench.py` | Command-line benchmark |
| REST | `init_bench_rest` | `restful/app.py` | Flask RESTful API |
| Batch | `vectordbbench batchcli --batch-config-file` | `cli/batch_cli.py` | Batch YAML execution |

## Key Classes & Abstractions

### VectorDB (backend/clients/api.py)
All database clients MUST implement this ABC:
- `__init__(dim, db_config, db_case_config, collection_name, drop_old)` - Initialize client
- `init()` (contextmanager) - Create/destroy connections safely
- `insert_embeddings(embeddings, metadata, labels_data)` - Insert vectors
- `search_embedding(query, k=100)` - Vector similarity search
- `optimize(data_size)` - Build index / optimize after insertion

### CaseType (backend/cases.py)
Enum defining all benchmark scenarios:
- `CapacityDim128/960` - Capacity/load tests
- `Performance768D100M/10M/1M` - Performance tests (various sizes)
- `Performance768D10M1P/99P` - Filtered search tests (1% / 99% filter rate)
- `StreamingPerformanceCase` - Streaming insert+search tests
- `LabelFilterPerformanceCase` - Label-based filtering tests
- `PerformanceCustomDataset` - User-defined dataset tests

### Dataset (backend/dataset.py)
Built-in datasets: SIFT, GIST, Cohere, OpenAI, LAION, Bioasq, Glove
- Downloaded from S3/AliyunOSS to `DATASET_LOCAL_DIR`
- Parquet format: `train.parquet`, `test.parquet`, `neighbors.parquet`

### BenchMarkRunner (interface.py)
Main orchestrator:
- `run(tasks, task_label)` - Submit benchmark tasks
- `get_results(result_dir)` - Retrieve historical results
- Uses `ProcessPoolExecutor` for async execution

## Data Flow

```
User Input (GUI/CLI)
    -> TaskConfig (models.py)
        -> Assembler.assemble_all() (assembler.py)
            -> TaskRunner with CaseRunner[] (task_runner.py)
                -> CaseRunner.run()
                    -> init_db()              # Create DB connection
                    -> dataset.prepare()      # Download dataset
                    -> _load_train_data()     # Concurrent insert
                    -> _optimize()            # Build index
                    -> _serial_search()       # Calculate recall/latency
                    -> _conc_search()         # Calculate QPS
                -> TestResult (models.py)
                    -> flush()                # Save JSON to results/
```

## Development Environment

This project uses `uv` for Python environment management. A `.venv` directory exists in the project root.

### Using the uv virtual environment

```bash
# Activate the venv
source .venv/bin/activate

# Or run commands directly via the venv Python
.venv/bin/python -m pytest tests/ -v

# Install dependencies with uv
uv pip install -e ".[test]"

# Install additional packages
uv pip install --python .venv/bin/python <package>
```

**Note:** The project requires Python 3.11+. The system `python` may be 3.10, so always use `.venv/bin/python` for running tests.

## Configuration

Environment variables (defined in `__init__.py`):
- `LOG_LEVEL` - Default: INFO
- `DATASET_SOURCE` - S3 or AliyunOSS
- `DATASET_LOCAL_DIR` - Default: /tmp/vectordb_bench/dataset
- `RESULTS_LOCAL_DIR` - Default: vectordb_bench/results
- `CONFIG_LOCAL_DIR` - Default: vectordb_bench/config-files
- `NUM_PER_BATCH` - Default: 100
- `NUM_CONCURRENCY` - Default: [1,5,10,20,30,40,60,80]
- `CONCURRENCY_DURATION` - Default: 30 (seconds)
- `DROP_OLD` - Default: True

## Adding a New Database Client

1. Create directory `backend/clients/mydb/`
2. Implement `config.py` with `MyDBConfig(DBConfig)` and `MyDBCaseConfig(DBCaseConfig)`
3. Implement `mydb.py` with `MyDB(VectorDB)` - 4 required methods
4. Register in `backend/clients/__init__.py` - add to DB enum and mappings
5. (Optional) Add CLI support with `cli.py` and register in `cli/vectordbbench.py`

## Common Tasks

### Run linting/formatting
```bash
make lint      # Check style
make format    # Auto-fix style
```

### Run a benchmark via CLI
```bash
# PgVector example
vectordbbench pgvectorhnsw \
  --host localhost --user-name postgres --password 'pass' \
  --db-name vectordb --case-type Performance768D10M \
  --m 16 --ef-construction 128 --ef-search 128

# Skip load, only search
vectordbbench pgvectorhnsw ... --skip-load --skip-drop-old

# Custom concurrency levels
vectordbbench pgvectorhnsw ... --num-concurrency 1,10,20,50
```

### Run GUI mode
```bash
init_bench
# or
python -m vectordb_bench
```

### Read results programmatically
```python
from vectordb_bench.interface import benchmark_runner
results = benchmark_runner.get_results()
```

## Important Notes

- **Python 3.11+ required**
- Uses `pydantic` v2 for all data models
- Uses `polars` (not pandas) for dataset reading
- Test cases support timeout controls (configurable per dataset size)
- Results are saved as JSON in `results/{db_name}/result_{date}_{label}_{db}.json`
- Supports custom datasets via Parquet format with specific column names
- Thread safety: set `thread_safe = False` on VectorDB subclass if client is not thread-safe

## Dependencies

Core: `click`, `streamlit`, `pydantic>=2.0`, `polars`, `plotly`, `pymilvus`, `scikit-learn`, `tqdm`

Optional (per database): `qdrant-client`, `pinecone`, `weaviate-client`, `elasticsearch`, `psycopg`, `redis`, `chromadb`, `pymongo`, etc.

## File Glossary

| File | Purpose |
|------|---------|
| `__init__.py` | Global configuration class, env var parsing |
| `interface.py` | BenchMarkRunner - main public API |
| `models.py` | TaskConfig, CaseConfig, TestResult, CaseResult |
| `metric.py` | Metric dataclass, recall/ndcg calculation |
| `backend/cases.py` | All test case definitions and CaseType enum |
| `backend/dataset.py` | Dataset definitions, DatasetManager, iterators |
| `backend/task_runner.py` | CaseRunner & TaskRunner execution |
| `backend/assembler.py` | TaskConfig -> CaseRunner assembly |
| `backend/clients/api.py` | VectorDB ABC, DBConfig, DBCaseConfig |
| `backend/clients/__init__.py` | DB enum, client registry mappings |
| `cli/cli.py` | CommonTypedDict, click parameter decorators |
| `frontend/vdbbench.py` | Streamlit app entry |
