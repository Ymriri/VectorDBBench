# AWS S3 Vectors 后端并发增强设计

**日期:** 2026-04-26
**作者:** brainstorming session
**状态:** 待 review

## 背景

`vectordb_bench/backend/clients/s3_vectors/` 已存在一个基础实现(PR #600),功能能跑但不具备生产可用的并发能力:

- 单线程串行批量插入,无连接池调优
- 无 ThrottlingException 自适应退避,limited 状态下会快速失败
- boto3 client 在 `init()` 里创建一次,但客户端配置未指定连接池上限,默认 `max_pool_connections=10` 在多 worker 场景下会成为瓶颈
- 文档错把类描述写成 "Wrapper around the Milvus" (复制粘贴遗留)

用户需要一个真正能在 `ConcurrentInsertRunner` 多 worker 模型下安全跑的实现,同时对 boto3 调优参数完全可配。

## 目标

1. 让 S3 Vectors 后端在 VectorDBBench 既有并发模型下安全运行
2. 暴露 boto3 关键调优参数(连接池 / 重试)给配置层和 CLI
3. 把 PutVectors 的 500 向量/调用 AWS 硬限制吸收到客户端,通过可配的 `insert_batch_size` 切片
4. 用 mock 单元测试守住关键并发与错误处理逻辑

## 非目标

- 不引入异步框架(aioboto3 / aiobotocore) — VectorDB ABC 是同步的,引入 async 桥接代价远超收益
- 不改 VectorDB 抽象类
- 不写真 AWS 集成测试(需要 AWS 账号 + 费用)
- 不引入 IAM role / profile 认证扩展(保持现有 access_key + secret_access_key 模式)
- 不在 `s3_vectors/` 目录新增 README.md(项目无先例,与既有客户端布局不一致)

## 架构

**单层并发模型**:`thread_safe=True`(基类默认值,不显式重写),让 `ConcurrentInsertRunner` 和 `MultiProcessingSearchRunner` 这两个项目标准 runner 来驱动并发。`S3Vectors` 实例在多个 worker 线程间共享,所有线程复用同一个 `boto3.client('s3vectors', ...)`(boto3 低阶 client 线程安全)。

**安全并发的两道兜底:**

1. **`botocore.config.Config`** 在 client 构造时注入:
   - `max_pool_connections` — urllib3 连接池上限(默认 50)
   - `retries={'mode': 'adaptive', 'max_attempts': 10}` — 自适应退避应对 ThrottlingException
2. **客户端切片** — `insert_embeddings` 把 runner 传入的大 batch 按 `insert_batch_size` (默认 100,AWS 上限 500)切成小片,每片一次 `put_vectors` 调用

**文件布局**(与现有完全一致,只改文件内容):

```
vectordb_bench/backend/clients/s3_vectors/
  ├── s3_vectors.py   # VectorDB 实现(重写)
  ├── config.py       # DBConfig + DBCaseConfig(扩展字段)
  └── cli.py          # click 入口(扩展参数)
tests/test_s3_vectors.py  # 新增:mock boto3 单元测试(项目内首例)
```

## 组件 & 接口

### `config.py` — `S3VectorsConfig`(DBConfig)

新增 4 个字段,全部带默认值,字段 docstring 解释 why:

```python
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
    Literal-typed so pydantic rejects invalid values at config-construction time
    (matches the CLI click choice)."""

    retry_max_attempts: int = 10
    """Total attempts including the first call. boto3 default is 3-5; raised to
    10 for benchmark stability under temporary throttling."""
```

`to_dict()` 同步输出**全部** 9 个字段(原有 5 + 新增 4)。`to_dict()` 是手写白名单,加字段必须同时改它,否则运行时拿不到。第 7 个测试用例守住这点。

### `config.py` — `S3VectorsIndexConfig`(DBCaseConfig)

不变。`parse_metric()` 仅识别 cosine / L2,遇未知 metric 抛 `ValueError`(让未来 AWS 新增 metric 在客户端显式失败,不静默乱传)。

### `s3_vectors.py` — `S3Vectors(VectorDB)`

公开方法保持现有 5 个签名(VectorDB 抽象接口要求):

| 方法 | 改动要点 |
|------|---------|
| `__init__` | 读取所有 boto3 调优字段;用 `botocore.config.Config(...)` 构造 client;drop_old + create_index 逻辑保持但用注入了 Config 的新 client |
| `init` (`@contextmanager`) | 创建长生命周期 client 存到 `self.client`,被多个 worker 线程共享;退出时 close |
| `insert_embeddings` | 按 `db_config.insert_batch_size` 切片;每片调一次 `put_vectors`;累计计数;遇异常返回 `(insert_count, e)` 让 runner 处理 |
| `search_embedding` | 不变(单次 `query_vectors`) |
| `prepare_filter` | 不变 |
| `optimize` | 保持空实现(S3 Vectors 无显式索引优化语义) |

**类级 docstring** 写并发模型:

```python
class S3Vectors(VectorDB):
    """AWS S3 Vectors backend for VectorDBBench.

    Concurrency model:
    - thread_safe=True (inherited from VectorDB base class).
    - The ConcurrentInsertRunner and MultiProcessingSearchRunner drive
      concurrency at the worker level. All workers share the same
      self.client built in init() — boto3's low-level client is thread-safe.
    - The urllib3 connection pool size is governed by
      db_config.max_pool_connections; size it >= 2 * worker count.
    - Adaptive retry with botocore handles ThrottlingException; we do NOT
      add a custom retry layer because that would collide with botocore's
      adaptive token bucket.
    - PutVectors is capped at 500 vectors/call by AWS; insert_embeddings
      chunks the runner's batch into db_config.insert_batch_size slices.
    """
```

**修正复制粘贴遗留:** 文件顶部 docstring `"""Wrapper around the Milvus..."""` 改为 `"""Wrapper around the AWS S3 Vectors service."""`。

### `cli.py` — 新增 click 选项

新增 4 个,带默认值:

- `--insert-batch-size` — int, default 100
- `--max-pool-connections` — int, default 50
- `--retry-mode` — choice ['legacy', 'standard', 'adaptive'], default 'adaptive'
- `--retry-max-attempts` — int, default 10

CLI 入口 `S3Vectors(...)` 把这些参数传到 `S3VectorsConfig(...)`。help 文案精简到一句话,详细 why 在 config 字段 docstring 里。

## 数据流

### 写入路径

```
ConcurrentInsertRunner worker thread (N workers, default min(cpu, 4))
      │
      ▼
insert_embeddings(embeddings[K], metadata[K], labels?)
      │
      │  for offset in range(0, K, insert_batch_size):
      │      slice_records = build_records(embeddings, metadata, labels, offset)
      │      self.client.put_vectors(... vectors=slice_records)   ← 共享 client
      │      insert_count += len(slice_records)
      │
      │  except ClientError as e:
      │      log.warning(...)
      │      return (insert_count, e)        ← 部分成功也返回计数
      ▼
Runner 决定重试 / 推进
```

关键点:
- N 个 worker 共享 `self.client`(boto3 client 线程安全)
- 连接池 = `max_pool_connections`(50),容纳 N=4 worker 各自串行发请求绰绰有余
- 切片大小由 `insert_batch_size` 控,不在客户端 hard-cap 500(留给 AWS 端报错,避免 AWS 调整限制后跟改)

### 搜索路径

不变,单次 `query_vectors`。filter 提前在 `prepare_filter` 写到 `self.filter`。

## 错误处理

| 错误来源 | 处理方 | 行为 |
|---------|-------|------|
| `ThrottlingException` / `ProvisionedThroughputExceededException` | botocore adaptive retry | 自动指数退避重试 ≤ `retry_max_attempts` 次;不上抛 |
| 其它 5xx / 网络抖动 | botocore 重试 | 同上 |
| 重试耗尽 | `s3_vectors.py` 捕获 `ClientError` | `log.warning` + 返回 `(insert_count, e)`;runner 决定 |
| Drop_old 时 index 不存在 | `__init__` | 静默跳过 delete,继续 create |
| 未知 `metric_type` | `parse_metric()` | 抛 `ValueError`,构造期就失败 |
| `assert len(embeddings) == len(metadata)` 不等 | `insert_embeddings` | `AssertionError`,runner 上抛(编程错误) |

**为什么不在 `insert_embeddings` 内加自定义重试:** botocore adaptive retry 已覆盖限流场景;自己再加一层会和 botocore 的 token bucket 冲突,叠加退避反而更慢。

## 测试

### 文件: `tests/test_s3_vectors.py`(项目内首个 mock boto3 测试)

测试**完全 mock `boto3.client`**,不连真 AWS,不引入 `moto`(项目现有 deps 没有,不为单后端加包)。

mock 用 `unittest.mock.patch('boto3.client')` 注入 `MagicMock`。测试默认随项目 pytest 跑,**不依赖任何环境变量**;CI 不需要改动。

### 7 个用例

| # | 测试名 | 验证点 |
|---|--------|--------|
| 1 | `test_client_built_with_botocore_config` | `boto3.client` 被调用时 `config=` 关键字参数包含 `max_pool_connections=50`、`retries={'mode':'adaptive', 'max_attempts':10}` |
| 2 | `test_insert_chunks_to_batch_size` | 输入 250 条 + `insert_batch_size=100` → mock client 收到 3 次 `put_vectors`(100/100/50) |
| 3 | `test_insert_returns_partial_count_on_error` | 第 2 次 `put_vectors` 抛 `ClientError` → 返回 `(100, exception)` |
| 4 | `test_drop_old_skips_when_index_absent` | `list_indexes` 返回空 → 不调 `delete_index`、直接调 `create_index` |
| 5 | `test_filter_translation` | NumGE → `{'id':{'$gte':N}}`;StrEqual → `{'label':val}`;其它 → `ValueError` |
| 6 | `test_thread_safe_attribute` | `S3Vectors.thread_safe is True`(防止后续维护者误改基类默认)|
| 7 | `test_config_to_dict_exposes_all_tuning_fields` | `S3VectorsConfig(...).to_dict()` 返回的 dict 包含 `insert_batch_size` / `max_pool_connections` / `retry_mode` / `retry_max_attempts` 4 个新键(防止白名单漏改)|

## 文档

不写独立文档/README。所有 why-explanation 入 docstring:

- `config.py` 字段 docstring → 每个调优参数语义 + 默认值理由 + AWS 限制(如 PutVectors ≤ 500)
- `s3_vectors.py` 类级 docstring → 并发模型说明
- `cli.py` click `help=` → 单行精简提示,指向字段 docstring

## 兼容性 & 迁移

- 既有 CLI 命令 `vectordbbench s3vectors` 仍可工作(新增字段都有默认值)
- 既有 `S3VectorsConfig(...)` Python 调用仍可工作
- DB 枚举 / frontend 图标映射 / `cli/vectordbbench.py` 注册 — 全部已存在,不动
- 行为变化:`max_pool_connections` 从 boto3 默认的 10 提升到 50;adaptive retry 替代 legacy。这两个变化对现有用户透明,只会让吞吐更稳

## 风险

- **mock 测试是项目内首例** — 后续维护者可能不熟悉 `unittest.mock.patch` 模式;靠 docstring 在测试文件顶部解释为什么不连真 AWS 来缓解
- **adaptive retry 跟 ConcurrentInsertRunner 的全局速率不感知** — 多 worker 同时被限流时各自退避,可能导致整体吞吐不稳定;不是阻塞性问题,后续可以考虑共享 token bucket(独立 PR)
