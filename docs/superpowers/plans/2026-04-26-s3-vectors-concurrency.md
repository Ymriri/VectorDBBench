# S3 Vectors 并发增强 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 把现有 `vectordb_bench/backend/clients/s3_vectors/` 后端改造成在 ConcurrentInsertRunner 多 worker 模型下安全运行,boto3 调优参数(连接池/重试/批大小)全部可配,并加 mock 单元测试守住关键逻辑。

**Architecture:** 单层并发模型 — `thread_safe=True`(基类默认),所有 worker 共享同一个 `boto3.client`(线程安全),`botocore.config.Config` 注入连接池上限和 adaptive retry,`insert_embeddings` 按 `insert_batch_size` 切片调 PutVectors。

**Tech Stack:** Python 3.11+, pydantic v2, boto3 / botocore, click, pytest + unittest.mock

**Spec:** `docs/superpowers/specs/2026-04-26-s3-vectors-concurrency-design.md`

---

## File Structure

| 文件 | 角色 | 改动 |
|------|------|------|
| `vectordb_bench/backend/clients/s3_vectors/config.py` | DBConfig + DBCaseConfig | 加 4 个调优字段(Literal 类型 + docstring),`to_dict()` 同步输出 |
| `vectordb_bench/backend/clients/s3_vectors/s3_vectors.py` | VectorDB 实现 | 修文件 docstring;加类 docstring 写并发模型;`__init__` 注入 `botocore.config.Config`;`init` 同样注入;`insert_embeddings` 按 `self.insert_batch_size` 切片;exception 收紧到 `ClientError`;`log.warning` 替代 `log.info` |
| `vectordb_bench/backend/clients/s3_vectors/cli.py` | click 入口 | 加 4 个 click 选项,wire 到 `S3VectorsConfig` |
| `tests/test_s3_vectors.py` | 单元测试(项目首个 mock boto3 测试) | 新建,7 个用例 |

## 测试运行约定

项目用 `uv` 管理 venv,根目录已有 `.venv`。所有 pytest 命令在仓库根目录 `/Users/ym/Desktop/git_vd/VectorDBBench` 下跑:

```bash
.venv/bin/pytest tests/test_s3_vectors.py -v
```

或先 `source .venv/bin/activate`,然后用 `pytest`。本 plan 的 Run 命令一律用 `.venv/bin/pytest` 形式以避免 shell 状态依赖。

---

## Task 1: 新增 boto3 调优字段到 S3VectorsConfig

**Files:**
- Modify: `vectordb_bench/backend/clients/s3_vectors/config.py:1-20`
- Create: `tests/test_s3_vectors.py`

**目标:** 给 `S3VectorsConfig` 加 4 个新字段(`insert_batch_size` / `max_pool_connections` / `retry_mode` / `retry_max_attempts`),`to_dict()` 同步输出。`retry_mode` 用 `Literal` 类型让 pydantic 在构造时校验。

- [ ] **Step 1: 创建测试文件,写第一个失败测试**

新建 `tests/test_s3_vectors.py`,内容:

```python
"""Mock-based unit tests for the AWS S3 Vectors backend.

These tests do not connect to AWS — boto3.client is patched with MagicMock.
Project precedent: this is the first mock-boto3 test file in tests/. Other AWS
backend tests (e.g. test_aws_opensearch_cli.py) only test helper functions.
The mock approach lets us verify chunking, retry config, and filter logic
without an AWS account or moto dependency.
"""

from unittest.mock import MagicMock, patch

import pytest
from pydantic import SecretStr


def _build_config(**overrides):
    from vectordb_bench.backend.clients.s3_vectors.config import S3VectorsConfig

    defaults = {
        "access_key_id": SecretStr("AKIA_FAKE"),
        "secret_access_key": SecretStr("secret_fake"),
        "bucket_name": "fake-bucket",
    }
    defaults.update(overrides)
    return S3VectorsConfig(**defaults)


def test_config_to_dict_exposes_all_tuning_fields():
    """to_dict() is a hand-maintained whitelist; this guards against forgetting
    to expose newly added fields to the runtime."""
    d = _build_config().to_dict()
    assert d["insert_batch_size"] == 100
    assert d["max_pool_connections"] == 50
    assert d["retry_mode"] == "adaptive"
    assert d["retry_max_attempts"] == 10
```

- [ ] **Step 2: 运行测试,确认失败**

Run: `.venv/bin/pytest tests/test_s3_vectors.py::test_config_to_dict_exposes_all_tuning_fields -v`

Expected: FAIL with `KeyError: 'insert_batch_size'` (因为 `to_dict()` 还没暴露新字段) 或 `ValidationError`(若 pydantic 这边没字段)。

- [ ] **Step 3: 改 `config.py`,加 4 个字段 + 改 `to_dict()`**

把 `vectordb_bench/backend/clients/s3_vectors/config.py` 整文件替换为:

```python
from typing import Literal

from pydantic import BaseModel, SecretStr

from ..api import DBCaseConfig, DBConfig, MetricType


class S3VectorsConfig(DBConfig):
    region_name: str = "us-west-2"
    access_key_id: SecretStr
    secret_access_key: SecretStr
    bucket_name: str
    index_name: str = "vdbbench-index"

    insert_batch_size: int = 100
    """PutVectors per-call batch size. AWS hard limit: 500. Larger means fewer API
    calls (cheaper) but higher per-call latency and memory; smaller means more API
    calls (more throttling risk) but more even latency. Recommended 100-500."""

    max_pool_connections: int = 50
    """urllib3 connection pool size for the boto3 client. Should be >= 2 * the
    ConcurrentInsertRunner worker count to avoid pool starvation. boto3 default
    is 10, which is too low for benchmark workloads."""

    retry_mode: Literal["legacy", "standard", "adaptive"] = "adaptive"
    """boto3 retry mode. 'adaptive' uses a token bucket to slow down on throttling;
    'standard' is fixed-attempt; 'legacy' is the boto3 v1 default. Adaptive
    recommended for S3 Vectors due to AWS service-level rate limits.
    Literal-typed so pydantic rejects invalid values at config-construction time."""

    retry_max_attempts: int = 10
    """Total attempts including the first call. boto3 default is 3-5; raised to
    10 for benchmark stability under temporary throttling."""

    def to_dict(self) -> dict:
        return {
            "region_name": self.region_name,
            "access_key_id": self.access_key_id.get_secret_value() if self.access_key_id else "",
            "secret_access_key": self.secret_access_key.get_secret_value() if self.secret_access_key else "",
            "bucket_name": self.bucket_name,
            "index_name": self.index_name,
            "insert_batch_size": self.insert_batch_size,
            "max_pool_connections": self.max_pool_connections,
            "retry_mode": self.retry_mode,
            "retry_max_attempts": self.retry_max_attempts,
        }


class S3VectorsIndexConfig(DBCaseConfig, BaseModel):
    """Base config for s3-vectors"""

    metric_type: MetricType | None = None
    data_type: str = "float32"

    def parse_metric(self) -> str:
        if self.metric_type == MetricType.COSINE:
            return "cosine"
        if self.metric_type == MetricType.L2:
            return "euclidean"
        msg = f"Unsupported metric type: {self.metric_type}"
        raise ValueError(msg)

    def index_param(self) -> dict:
        return {}

    def search_param(self) -> dict:
        return {}
```

- [ ] **Step 4: 运行测试,确认通过**

Run: `.venv/bin/pytest tests/test_s3_vectors.py::test_config_to_dict_exposes_all_tuning_fields -v`

Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add vectordb_bench/backend/clients/s3_vectors/config.py tests/test_s3_vectors.py
git commit -m "$(cat <<'EOF'
feat(s3_vectors): add boto3 tuning fields to config

Add insert_batch_size, max_pool_connections, retry_mode, and
retry_max_attempts to S3VectorsConfig with sensible benchmark defaults
(100, 50, adaptive, 10). retry_mode is Literal-typed so invalid values
fail at pydantic construction time. to_dict() exposes all four fields.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 2: 暴露调优字段到 CLI

**Files:**
- Modify: `vectordb_bench/backend/clients/s3_vectors/cli.py:18-67`

**目标:** 给 `S3VectorsTypedDict` 加 4 个 click 选项,在 `S3Vectors()` entry 里把它们传到 `S3VectorsConfig`。CLI 是 glue code,不写专门的测试 — 通过 `--help` 输出和 dry-import 验证。

- [ ] **Step 1: 改 `cli.py` 的 `S3VectorsTypedDict` 和 `S3Vectors()`**

把 `vectordb_bench/backend/clients/s3_vectors/cli.py` 整文件替换为:

```python
from typing import Annotated, TypedDict, Unpack

import click
from pydantic import SecretStr

from ....cli.cli import (
    CommonTypedDict,
    cli,
    click_parameter_decorators_from_typed_dict,
    get_custom_case_config,
    run,
)
from .. import DB
from ..api import MetricType
from .config import S3VectorsIndexConfig


class S3VectorsTypedDict(TypedDict):
    region_name: Annotated[
        str, click.option("--region", type=str, help="AWS region for S3 bucket (eg. us-east-1)", default="us-east-1")
    ]
    access_key_id: Annotated[str, click.option("--access_key_id", type=str, help="AWS access key ID", required=True)]
    secret_access_key: Annotated[
        str, click.option("--secret_access_key", type=str, help="AWS secret access key", required=True)
    ]

    bucket: Annotated[str, click.option("--bucket", type=str, help="S3 bucket name", required=True)]
    index: Annotated[str, click.option("--index", type=str, help="Unique vector index name", default="vdbbench-index")]

    metric: Annotated[
        str,
        click.option(
            "--metric",
            type=str,
            help="Distance metric for vector similarity (e.g., 'cosine', 'euclidean').",
            default=None,
        ),
    ]

    insert_batch_size: Annotated[
        int,
        click.option(
            "--insert-batch-size",
            type=int,
            help="PutVectors batch size; AWS hard limit 500 per call",
            default=100,
            show_default=True,
        ),
    ]
    max_pool_connections: Annotated[
        int,
        click.option(
            "--max-pool-connections",
            type=int,
            help="urllib3 connection pool size; should be >= 2x ConcurrentInsertRunner worker count",
            default=50,
            show_default=True,
        ),
    ]
    retry_mode: Annotated[
        str,
        click.option(
            "--retry-mode",
            type=click.Choice(["legacy", "standard", "adaptive"]),
            help="boto3 retry mode; adaptive recommended for throttling resilience",
            default="adaptive",
            show_default=True,
        ),
    ]
    retry_max_attempts: Annotated[
        int,
        click.option(
            "--retry-max-attempts",
            type=int,
            help="boto3 retry total attempt count (including the first call)",
            default=10,
            show_default=True,
        ),
    ]


class S3VectorsIndexTypedDict(CommonTypedDict, S3VectorsTypedDict): ...


@cli.command()
@click_parameter_decorators_from_typed_dict(S3VectorsIndexTypedDict)
def S3Vectors(**parameters: Unpack[S3VectorsIndexTypedDict]):
    from .config import S3VectorsConfig

    parameters["custom_case"] = get_custom_case_config(parameters)
    run(
        db=DB.S3Vectors,
        db_config=S3VectorsConfig(
            region_name=parameters["region"],
            access_key_id=SecretStr(parameters["access_key_id"]),
            secret_access_key=SecretStr(parameters["secret_access_key"]),
            bucket_name=parameters["bucket"],
            index_name=parameters["index"] if parameters["index"] else "vdbbench-index",
            insert_batch_size=parameters["insert_batch_size"],
            max_pool_connections=parameters["max_pool_connections"],
            retry_mode=parameters["retry_mode"],
            retry_max_attempts=parameters["retry_max_attempts"],
        ),
        db_case_config=S3VectorsIndexConfig(
            metric_type=(
                MetricType.COSINE
                if parameters["metric"] == "cosine"
                else MetricType.L2 if parameters["metric"] == "l2" else None
            )
        ),
        **parameters,
    )
```

- [ ] **Step 2: 验证 CLI 加载并显示新选项**

Run: `.venv/bin/python -c "from vectordb_bench.cli.vectordbbench import cli; cli(['s3vectors', '--help'], standalone_mode=False)" 2>&1 | grep -E "insert-batch-size|max-pool-connections|retry-mode|retry-max-attempts"`

Expected: 4 行,每行一个新选项的 help 文案。

- [ ] **Step 3: Commit**

```bash
git add vectordb_bench/backend/clients/s3_vectors/cli.py
git commit -m "$(cat <<'EOF'
feat(s3_vectors): expose boto3 tuning fields to CLI

Add --insert-batch-size, --max-pool-connections, --retry-mode (Choice),
and --retry-max-attempts options. Wire them through to S3VectorsConfig.
Defaults match config-side defaults (100, 50, adaptive, 10).

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 3: 在 `__init__` 注入 botocore Config

**Files:**
- Modify: `vectordb_bench/backend/clients/s3_vectors/s3_vectors.py:1-74`
- Modify: `tests/test_s3_vectors.py`

**目标:** `__init__` 阶段构造 `botocore.config.Config` 并传给 `boto3.client(..., config=...)`,把它存到 `self._botocore_config` 供 `init()` 复用。drop_old/create_index 用同一个临时 client。

- [ ] **Step 1: 在测试文件追加测试**

把以下加到 `tests/test_s3_vectors.py` 末尾:

```python
def _default_db_and_case():
    """Build a default (db_config dict, S3VectorsIndexConfig) pair for tests
    that need to construct an S3Vectors instance. Tests still must patch
    boto3 themselves — this helper only assembles the config inputs."""
    from vectordb_bench.backend.clients.api import MetricType
    from vectordb_bench.backend.clients.s3_vectors.config import S3VectorsIndexConfig

    db_config = _build_config().to_dict()
    case_config = S3VectorsIndexConfig(metric_type=MetricType.COSINE)
    return db_config, case_config


def test_client_built_with_botocore_config():
    """boto3.client must receive a botocore Config with our tuning values so
    urllib3 pool size and adaptive retry actually take effect."""
    from vectordb_bench.backend.clients.s3_vectors import s3_vectors as mod

    db_config, case_config = _default_db_and_case()

    with patch.object(mod, "boto3") as mock_boto3:
        fake_client = MagicMock()
        mock_boto3.client.return_value = fake_client

        mod.S3Vectors(
            dim=4,
            db_config=db_config,
            db_case_config=case_config,
            drop_old=False,
        )

        assert mock_boto3.client.called
        call_kwargs = mock_boto3.client.call_args.kwargs
        assert "config" in call_kwargs, "boto3.client must be called with config="
        cfg = call_kwargs["config"]
        assert cfg.max_pool_connections == 50
        assert cfg.retries == {"mode": "adaptive", "max_attempts": 10}
```

- [ ] **Step 2: 运行测试,确认失败**

Run: `.venv/bin/pytest tests/test_s3_vectors.py::test_client_built_with_botocore_config -v`

Expected: FAIL with `AssertionError: boto3.client must be called with config=` (现有代码不传 config)。

- [ ] **Step 3: 改 `s3_vectors.py` 的 `__init__` 注入 Config**

整文件替换 `vectordb_bench/backend/clients/s3_vectors/s3_vectors.py`,**只换 1-74 行的部分**(其它部分本 task 不动,Task 4-6 再改):

```python
"""Wrapper around the AWS S3 Vectors service."""

import logging
from collections.abc import Iterable
from contextlib import contextmanager

import boto3
from botocore.config import Config
from botocore.exceptions import ClientError

from vectordb_bench.backend.filter import Filter, FilterOp

from ..api import VectorDB
from .config import S3VectorsIndexConfig

log = logging.getLogger(__name__)


class S3Vectors(VectorDB):
    """AWS S3 Vectors backend for VectorDBBench.

    Concurrency model:
    - thread_safe=True (inherited from VectorDB base class).
    - The ConcurrentInsertRunner and MultiProcessingSearchRunner drive
      concurrency at the worker level. All workers share the same
      self.client built in init() — boto3's low-level client is thread-safe.
    - The urllib3 connection pool size is governed by
      db_config["max_pool_connections"]; size it >= 2 * worker count.
    - Adaptive retry with botocore handles ThrottlingException; we do NOT
      add a custom retry layer because that would collide with botocore's
      adaptive token bucket.
    - PutVectors is capped at 500 vectors/call by AWS; insert_embeddings
      chunks the runner's batch into db_config["insert_batch_size"] slices.
    """

    supported_filter_types: list[FilterOp] = [
        FilterOp.NonFilter,
        FilterOp.NumGE,
        FilterOp.StrEqual,
    ]

    def __init__(
        self,
        dim: int,
        db_config: dict,
        db_case_config: S3VectorsIndexConfig,
        drop_old: bool = False,
        with_scalar_labels: bool = False,
        **kwargs,
    ):
        """Initialize wrapper around the s3-vectors client."""
        self.db_config = db_config
        self.case_config = db_case_config
        self.with_scalar_labels = with_scalar_labels

        self.insert_batch_size = self.db_config["insert_batch_size"]

        self._scalar_id_field = "id"
        self._scalar_label_field = "label"
        self._vector_field = "vector"

        self.region_name = self.db_config.get("region_name")
        self.access_key_id = self.db_config.get("access_key_id")
        self.secret_access_key = self.db_config.get("secret_access_key")
        self.bucket_name = self.db_config.get("bucket_name")
        self.index_name = self.db_config.get("index_name")

        self._botocore_config = Config(
            max_pool_connections=self.db_config["max_pool_connections"],
            retries={
                "mode": self.db_config["retry_mode"],
                "max_attempts": self.db_config["retry_max_attempts"],
            },
        )

        setup_client = boto3.client(
            service_name="s3vectors",
            region_name=self.region_name,
            aws_access_key_id=self.access_key_id,
            aws_secret_access_key=self.secret_access_key,
            config=self._botocore_config,
        )

        if drop_old:
            # delete old index if exists
            response = setup_client.list_indexes(vectorBucketName=self.bucket_name)
            index_names = [index["indexName"] for index in response["indexes"]]
            if self.index_name in index_names:
                log.info(f"drop old index: {self.index_name}")
                setup_client.delete_index(vectorBucketName=self.bucket_name, indexName=self.index_name)

            # create the index
            setup_client.create_index(
                vectorBucketName=self.bucket_name,
                indexName=self.index_name,
                dataType=self.case_config.data_type,
                dimension=dim,
                distanceMetric=self.case_config.parse_metric(),
            )

        setup_client.close()

    @contextmanager
    def init(self):
        """Yield with a long-lived boto3 client shared by all worker threads.

        boto3's low-level client is thread-safe; the connection pool size and
        retry behavior are set via self._botocore_config.
        """
        self.client = boto3.client(
            service_name="s3vectors",
            region_name=self.region_name,
            aws_access_key_id=self.access_key_id,
            aws_secret_access_key=self.secret_access_key,
            config=self._botocore_config,
        )

        yield
        self.client.close()

    def optimize(self, **kwargs):
        return

    def need_normalize_cosine(self) -> bool:
        """Wheather this database need to normalize dataset to support COSINE"""
        return False

    def insert_embeddings(
        self,
        embeddings: Iterable[list[float]],
        metadata: list[int],
        labels_data: list[str] | None = None,
        **kwargs,
    ) -> tuple[int, Exception | None]:
        """Insert embeddings into S3 Vectors via PutVectors.

        Chunks the input into self.insert_batch_size slices (≤ 500, AWS hard
        limit). On error returns (count_so_far, exception); the runner decides
        whether to retry the remainder.
        """
        assert self.client is not None
        assert len(embeddings) == len(metadata)
        insert_count = 0
        try:
            for batch_start_offset in range(0, len(embeddings), self.insert_batch_size):
                batch_end_offset = min(batch_start_offset + self.insert_batch_size, len(embeddings))
                insert_data = [
                    {
                        "key": str(metadata[i]),
                        "data": {self.case_config.data_type: embeddings[i]},
                        "metadata": (
                            {self._scalar_label_field: labels_data[i], self._scalar_id_field: metadata[i]}
                            if self.with_scalar_labels
                            else {self._scalar_id_field: metadata[i]}
                        ),
                    }
                    for i in range(batch_start_offset, batch_end_offset)
                ]
                self.client.put_vectors(
                    vectorBucketName=self.bucket_name,
                    indexName=self.index_name,
                    vectors=insert_data,
                )
                insert_count += len(insert_data)
        except ClientError as e:
            log.warning(f"S3 Vectors put_vectors failed after {insert_count} inserts: {e}")
            return insert_count, e
        return insert_count, None

    def prepare_filter(self, filters: Filter):
        if filters.type == FilterOp.NonFilter:
            self.filter = None
        elif filters.type == FilterOp.NumGE:
            self.filter = {self._scalar_id_field: {"$gte": filters.int_value}}
        elif filters.type == FilterOp.StrEqual:
            self.filter = {self._scalar_label_field: filters.label_value}
        else:
            msg = f"Not support Filter for S3Vectors - {filters}"
            raise ValueError(msg)

    def search_embedding(
        self,
        query: list[float],
        k: int = 100,
        timeout: int | None = None,
    ) -> list[int]:
        """Perform a search on a query embedding and return results."""
        assert self.client is not None

        # Perform the search.
        res = self.client.query_vectors(
            vectorBucketName=self.bucket_name,
            indexName=self.index_name,
            queryVector={"float32": query},
            topK=k,
            filter=self.filter,
            returnDistance=False,
            returnMetadata=False,
        )

        # Organize results.
        return [int(result["key"]) for result in res["vectors"]]
```

注:这一步直接把 `s3_vectors.py` 整个文件替换成最终版本(包含 Tasks 3-6 的所有改动)。Task 4-6 之后只是验证已经在这一步落到位的代码。文件级整体替换比 Task-by-Task 局部 patch 更安全 — 重读完整文件比追多个 patch 容易出错。

- [ ] **Step 4: 运行测试,确认通过**

Run: `.venv/bin/pytest tests/test_s3_vectors.py::test_client_built_with_botocore_config -v`

Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add vectordb_bench/backend/clients/s3_vectors/s3_vectors.py tests/test_s3_vectors.py
git commit -m "$(cat <<'EOF'
feat(s3_vectors): inject botocore Config for safe concurrent access

Build a botocore.config.Config from db_config tuning fields and pass it
to both the setup boto3 client (used for drop_old/create_index) and the
long-lived client created in init() (shared by all worker threads).

Also: fix file-level docstring (was Milvus copy-paste); add class-level
docstring documenting the concurrency model; switch insert_embeddings
exception handler from generic Exception to botocore ClientError; raise
log level from info to warning on insert failure; replace hard-coded
self.batch_size = 500 with self.insert_batch_size from db_config.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 4: 验证 insert 切片行为(test #2)

**Files:**
- Modify: `tests/test_s3_vectors.py`

**目标:** 验证 `insert_embeddings(250 条) + insert_batch_size=100` → mock client 收到 3 次 put_vectors,size 100/100/50。代码已经在 Task 3 落地,这里只补测试。

- [ ] **Step 1: 在测试文件追加测试**

追加到 `tests/test_s3_vectors.py` 末尾:

```python
def test_insert_chunks_to_batch_size():
    """250 records + insert_batch_size=100 → exactly 3 put_vectors calls
    with sizes 100, 100, 50."""
    from vectordb_bench.backend.clients.s3_vectors import s3_vectors as mod

    db_config, case_config = _default_db_and_case()

    with patch.object(mod, "boto3") as mock_boto3:
        fake_client = MagicMock()
        mock_boto3.client.return_value = fake_client

        db = mod.S3Vectors(
            dim=4,
            db_config=db_config,
            db_case_config=case_config,
            drop_old=False,
        )
        with db.init():
            count, err = db.insert_embeddings(
                embeddings=[[0.1, 0.2, 0.3, 0.4]] * 250,
                metadata=list(range(250)),
            )

        assert count == 250
        assert err is None
        assert fake_client.put_vectors.call_count == 3
        sizes = [len(call.kwargs["vectors"]) for call in fake_client.put_vectors.call_args_list]
        assert sizes == [100, 100, 50]
```

- [ ] **Step 2: 运行测试,确认通过(Task 3 实现已经覆盖)**

Run: `.venv/bin/pytest tests/test_s3_vectors.py::test_insert_chunks_to_batch_size -v`

Expected: PASS

- [ ] **Step 3: Commit**

```bash
git add tests/test_s3_vectors.py
git commit -m "$(cat <<'EOF'
test(s3_vectors): verify insert_embeddings chunks per insert_batch_size

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 5: 验证 insert 部分成功语义(test #3)

**Files:**
- Modify: `tests/test_s3_vectors.py`

**目标:** 第 2 次 put_vectors 抛 ClientError → `insert_embeddings` 返回 `(100, exception)`,即首批已成功的计数 + 异常对象。

- [ ] **Step 1: 在测试文件追加测试**

追加到 `tests/test_s3_vectors.py` 末尾:

```python
def test_insert_returns_partial_count_on_error():
    """Second put_vectors raises ClientError → insert_embeddings returns
    (count_from_first_batch, exception). Validates the contract that
    ConcurrentInsertRunner relies on for partial-success accounting."""
    from botocore.exceptions import ClientError

    from vectordb_bench.backend.clients.s3_vectors import s3_vectors as mod

    db_config, case_config = _default_db_and_case()

    err = ClientError(
        error_response={"Error": {"Code": "ThrottlingException", "Message": "throttled"}},
        operation_name="PutVectors",
    )

    with patch.object(mod, "boto3") as mock_boto3:
        fake_client = MagicMock()
        # First call OK (returns mock); second call raises.
        fake_client.put_vectors.side_effect = [MagicMock(), err]
        mock_boto3.client.return_value = fake_client

        db = mod.S3Vectors(
            dim=4,
            db_config=db_config,
            db_case_config=case_config,
            drop_old=False,
        )
        with db.init():
            count, returned_err = db.insert_embeddings(
                embeddings=[[0.1, 0.2, 0.3, 0.4]] * 250,
                metadata=list(range(250)),
            )

        assert count == 100, "First batch (100 records) should have committed"
        assert returned_err is err, "The exact ClientError instance should be returned"
```

- [ ] **Step 2: 运行测试,确认通过**

Run: `.venv/bin/pytest tests/test_s3_vectors.py::test_insert_returns_partial_count_on_error -v`

Expected: PASS

- [ ] **Step 3: Commit**

```bash
git add tests/test_s3_vectors.py
git commit -m "$(cat <<'EOF'
test(s3_vectors): verify partial-count + exception return on insert error

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 6: 三个回归保护测试(tests #4 #5 #6)

**Files:**
- Modify: `tests/test_s3_vectors.py`

**目标:** 守住已有但容易被未来 PR 误改的行为 — drop_old 索引不存在时跳过 delete、filter 翻译表、`thread_safe=True`。

- [ ] **Step 1: 在测试文件追加 3 个测试**

追加到 `tests/test_s3_vectors.py` 末尾:

```python
def test_drop_old_skips_when_index_absent():
    """list_indexes returns empty → __init__ must not call delete_index,
    and must still call create_index. Guards against accidentally adding
    an unconditional delete_index that 404s on a fresh bucket."""
    from vectordb_bench.backend.clients.s3_vectors import s3_vectors as mod

    db_config, case_config = _default_db_and_case()

    with patch.object(mod, "boto3") as mock_boto3:
        fake_client = MagicMock()
        fake_client.list_indexes.return_value = {"indexes": []}
        mock_boto3.client.return_value = fake_client

        mod.S3Vectors(
            dim=4,
            db_config=db_config,
            db_case_config=case_config,
            drop_old=True,
        )

        fake_client.delete_index.assert_not_called()
        fake_client.create_index.assert_called_once()


def test_filter_translation():
    """Verify each FilterOp branch in prepare_filter."""
    from vectordb_bench.backend.clients.s3_vectors import s3_vectors as mod
    from vectordb_bench.backend.filter import Filter, FilterOp

    db_config, case_config = _default_db_and_case()

    with patch.object(mod, "boto3") as mock_boto3:
        mock_boto3.client.return_value = MagicMock()
        db = mod.S3Vectors(
            dim=4,
            db_config=db_config,
            db_case_config=case_config,
            drop_old=False,
        )

        f_none = Filter(type=FilterOp.NonFilter)
        db.prepare_filter(f_none)
        assert db.filter is None

        f_num = Filter(type=FilterOp.NumGE, int_value=42)
        db.prepare_filter(f_num)
        assert db.filter == {"id": {"$gte": 42}}

        f_str = Filter(type=FilterOp.StrEqual, label_value="cat")
        db.prepare_filter(f_str)
        assert db.filter == {"label": "cat"}


def test_thread_safe_attribute():
    """Defense against future maintainers flipping thread_safe to False —
    the implementation deliberately shares one boto3 client across threads
    and relies on this attribute being True."""
    from vectordb_bench.backend.clients.s3_vectors.s3_vectors import S3Vectors

    assert S3Vectors.thread_safe is True
```

- [ ] **Step 2: 运行 3 个测试,确认全过**

Run: `.venv/bin/pytest tests/test_s3_vectors.py::test_drop_old_skips_when_index_absent tests/test_s3_vectors.py::test_filter_translation tests/test_s3_vectors.py::test_thread_safe_attribute -v`

Expected: 3 PASS

- [ ] **Step 3: 跑整个测试文件,确认 7 个测试全过**

Run: `.venv/bin/pytest tests/test_s3_vectors.py -v`

Expected: 7 passed

- [ ] **Step 4: Commit**

```bash
git add tests/test_s3_vectors.py
git commit -m "$(cat <<'EOF'
test(s3_vectors): add regression guards for drop_old, filter, thread_safe

drop_old must skip delete_index when the index doesn't exist; filter
translation covers the three supported FilterOp branches; thread_safe
must remain True so ConcurrentInsertRunner shares the boto3 client
across worker threads.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 7: 收尾 — 跑全套项目测试 + lint

**目标:** 确认本次改动没有破坏既有测试、ruff/black 不报问题。

- [ ] **Step 1: 跑整个 tests/ 目录**

Run: `.venv/bin/pytest tests/ -v --ignore=tests/test_pgvector.py -m "not integration"`

Expected: 全部测试 PASS,跳过 integration 标记的(它们需要真服务)。

如果有 fail,定位是不是本次改动引入(本次只动 `s3_vectors/` 和 `tests/test_s3_vectors.py`)。

- [ ] **Step 2: 跑 ruff lint**

Run: `.venv/bin/ruff check vectordb_bench/backend/clients/s3_vectors/ tests/test_s3_vectors.py`

Expected: All checks passed.

如果有问题,按 ruff 提示修(常见:import 顺序、未用 import、行长度)。

- [ ] **Step 3: 跑 black 格式检查**

Run: `.venv/bin/black --check vectordb_bench/backend/clients/s3_vectors/ tests/test_s3_vectors.py`

Expected: would reformat 0 files。

如果有 reformat 需求:`.venv/bin/black vectordb_bench/backend/clients/s3_vectors/ tests/test_s3_vectors.py`,然后 amend 上一个 commit 或追加一笔 `style(s3_vectors): black format`。

- [ ] **Step 4: 最终验证,空 commit 不需要**

Run: `git log --oneline -10`

Expected: 看到 6 笔本次新增 commit(Task 1-6),HEAD 是最后一个 test regression commit。

---

## 备注

- **不开新文件、不拆模块**:全部改动落在已有的 3 个 s3_vectors/ 文件 + 1 个新 tests/ 文件,跟 spec 的 Non-Goals 对齐。
- **TDD 顺序**:每个 Task 都是「写失败测试 → 跑确认失败 → 改实现 → 跑确认通过 → commit」(Task 1/3 严格 TDD;Task 4/5/6 因为代码已在 Task 3 落地,变成「写测试 → 验证通过 → commit」回归保护模式)。
- **commit 粒度**:6 笔小 commit,每笔功能内聚,方便 reviewer。
- **不引入 moto**:全靠 `unittest.mock.patch`,不增加 deps。
