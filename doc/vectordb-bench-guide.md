# VectorDBBench 项目完全指南

> 本文档基于 VectorDBBench 代码库生成，涵盖项目架构、模块说明、核心流程、使用方式等内容，方便开发者快速理解和学习该项目。

---

## 1. 项目概述

### 1.1 什么是 VectorDBBench

**VectorDBBench（VDBBench）** 是一个开源的向量数据库基准测试工具，由 Zilliz（Milvus 背后的公司）发起和维护。它不仅提供主流向量数据库和云服务的基准测试结果，更是一套完整的性能与成本效益对比工具。

**核心能力：**
- **多数据库支持**：支持 Milvus、Zilliz Cloud、Elasticsearch、Pinecone、Qdrant、Weaviate、PgVector、Redis、Chroma、MongoDB 等 30+ 种向量数据库
- **多种测试场景**：容量测试、搜索性能测试、过滤搜索测试、流式性能测试
- **可视化界面**：基于 Streamlit 的 Web UI，直观展示测试结果
- **命令行支持**：完整的 CLI 工具，支持脚本化和自动化测试
- **标准数据集**：内置 SIFT、GIST、Cohere、OpenAI、LAION 等真实生产环境数据集

### 1.2 项目入口点

| 入口方式 | 命令 | 说明 |
|---------|------|------|
| **GUI 模式** | `init_bench` 或 `python -m vectordb_bench` | 启动 Streamlit Web 界面 |
| **CLI 模式** | `vectordbbench [OPTIONS] COMMAND [ARGS]...` | 命令行执行特定数据库测试 |
| **RESTful 模式** | `init_bench_rest` | 启动 Flask RESTful 服务 |
| **批量配置** | `vectordbbench batchcli --batch-config-file <file>` | 批量执行 YAML 配置 |

---

## 2. 整体架构

### 2.1 架构图

```
┌─────────────────────────────────────────────────────────────┐
│                        用户交互层                             │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐      │
│  │ Streamlit UI │  │  CLI Tool    │  │ RESTful API  │      │
│  │  (frontend/) │  │   (cli/)     │  │  (restful/)  │      │
│  └──────┬───────┘  └──────┬───────┘  └──────┬───────┘      │
└─────────┼─────────────────┼─────────────────┼──────────────┘
          │                 │                 │
          └─────────────────┼─────────────────┘
                            ▼
┌─────────────────────────────────────────────────────────────┐
│                      核心业务逻辑层                           │
│  ┌─────────────────────────────────────────────────────┐   │
│  │              BenchMarkRunner (interface.py)          │   │
│  │         - 任务调度、异步执行、结果收集                  │   │
│  └─────────────────────────────────────────────────────┘   │
│                            │                                │
│  ┌─────────────────────────┼─────────────────────────┐     │
│  ▼                         ▼                         ▼     │
│ ┌────────────┐      ┌────────────┐      ┌────────────┐   │
│ │TaskRunner  │      │ CaseRunner │      │  Metric    │   │
│ │(任务编排)   │      │(用例执行)   │      │(指标计算)   │   │
│ └─────┬──────┘      └─────┬──────┘      └────────────┘   │
│       │                   │                                │
│       ▼                   ▼                                │
│ ┌────────────┐      ┌────────────┐                        │
│ │  Assembler │      │  Runners   │                        │
│ │(组装用例)   │      │(执行引擎)   │                        │
│ └────────────┘      └─────┬──────┘                        │
│                           │                                │
└───────────────────────────┼────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────┐
│                       数据库适配层                            │
│  ┌─────────────────────────────────────────────────────┐   │
│  │           VectorDB 抽象接口 (clients/api.py)         │   │
│  │  - insert_embeddings()                              │   │
│  │  - search_embedding()                               │   │
│  │  - optimize()                                       │   │
│  └─────────────────────────────────────────────────────┘   │
│                            │                                │
│  ┌─────────┬─────────┬─────┴────┬─────────┬─────────┐     │
│  ▼         ▼         ▼          ▼         ▼         ▼     │
│ Milvus  Elastic  PgVector   Pinecone  Qdrant   Weaviate  │
│  ...     ...       ...        ...      ...       ...     │
└─────────────────────────────────────────────────────────────┘
          │
          ▼
┌─────────────────────────────────────────────────────────────┐
│                       数据与存储层                            │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐      │
│  │  Dataset     │  │ DataSource   │  │   Results    │      │
│  │(数据集管理)   │  │ (S3/OSS)     │  │ (结果存储)    │      │
│  └──────────────┘  └──────────────┘  └──────────────┘      │
└─────────────────────────────────────────────────────────────┘
```

### 2.2 核心设计思想

1. **插件化数据库适配**：通过 `VectorDB` 抽象类，任何新数据库只需实现 4 个核心方法即可接入
2. **配置驱动**：所有测试参数通过 Pydantic Model 进行类型安全校验
3. **多进程并发**：搜索测试使用多进程模型模拟真实并发场景
4. **异步任务队列**：使用 `ProcessPoolExecutor` 实现非阻塞的任务执行
5. **结果持久化**：测试结果自动保存为 JSON，支持历史对比

---

## 3. 目录结构详解

```
VectorDBBench/
├── pyproject.toml              # 项目配置、依赖、entry points
├── install.py                  # 安装脚本
├── Makefile                    # 常用命令（lint、format）
├── Dockerfile                  # 容器化部署
│
├── vectordb_bench/             # ===== 核心代码包 =====
│   ├── __init__.py             # 全局配置 (config class)
│   ├── __main__.py             # GUI 入口: 启动 Streamlit
│   ├── interface.py            # 核心接口: BenchMarkRunner
│   ├── models.py               # 数据模型: TaskConfig, CaseResult, TestResult
│   ├── metric.py               # 指标定义: Metric, recall, ndcg 计算
│   ├── base.py                 # 基础模型基类
│   │
│   ├── backend/                # ===== 后端逻辑 =====
│   │   ├── cases.py            # 测试用例定义 (CaseType, CapacityCase, PerformanceCase)
│   │   ├── dataset.py          # 数据集管理 (Dataset, DatasetManager)
│   │   ├── data_source.py      # 数据源 (S3, AliyunOSS 下载)
│   │   ├── filter.py           # 过滤条件 (IntFilter, LabelFilter)
│   │   ├── assembler.py        # 任务组装器
│   │   ├── task_runner.py      # 任务/用例执行器
│   │   ├── result_collector.py # 结果收集器
│   │   ├── utils.py            # 工具函数
│   │   │
│   │   ├── clients/            # ===== 数据库客户端 =====
│   │   │   ├── api.py          # VectorDB 抽象接口 + DBConfig + DBCaseConfig
│   │   │   ├── __init__.py     # DB 枚举 + 客户端注册表
│   │   │   ├── milvus/         # Milvus 客户端实现
│   │   │   ├── zilliz_cloud/   # Zilliz Cloud 客户端
│   │   │   ├── elastic_cloud/  # Elastic Cloud 客户端
│   │   │   ├── pgvector/       # PgVector 客户端
│   │   │   ├── pinecone/       # Pinecone 客户端
│   │   │   ├── qdrant_cloud/   # Qdrant Cloud 客户端
│   │   │   ├── weaviate_cloud/ # Weaviate Cloud 客户端
│   │   │   ├── redis/          # Redis 客户端
│   │   │   ├── chroma/         # Chroma 客户端
│   │   │   ├── mongodb/        # MongoDB 客户端
│   │   │   ├── oceanbase/      # OceanBase 客户端
│   │   │   ├── doris/          # Doris 客户端
│   │   │   └── ... (共 40+ 个数据库)
│   │   │
│   │   └── runner/             # ===== 执行引擎 =====
│   │       ├── serial_runner.py      # 串行插入/搜索
│   │       ├── concurrent_runner.py  # 并发插入
│   │       ├── mp_runner.py          # 多进程并发搜索
│   │       ├── read_write_runner.py  # 读写混合 (流式)
│   │       ├── rate_runner.py        # 速率控制
│   │       └── executor.py           # 执行器工具
│   │
│   ├── cli/                    # ===== 命令行工具 =====
│   │   ├── vectordbbench.py    # CLI 主入口
│   │   ├── cli.py              # 通用 CLI 框架 + CommonTypedDict
│   │   └── batch_cli.py        # 批量执行 YAML 配置
│   │
│   ├── frontend/               # ===== Streamlit 前端 =====
│   │   ├── vdbbench.py         # Streamlit 主入口
│   │   ├── pages/              # 各页面
│   │   │   ├── run_test.py     # 运行测试页面
│   │   │   ├── results.py      # 结果展示页面
│   │   │   ├── qps_recall.py   # QPS-Recall 曲线
│   │   │   ├── tables.py       # 表格视图
│   │   │   ├── custom.py       # 自定义数据集
│   │   │   └── ...
│   │   ├── components/         # UI 组件
│   │   └── config/             # 前端配置
│   │
│   ├── results/                # 测试结果存储目录
│   ├── config-files/           # 配置文件示例
│   ├── custom/                 # 自定义用例配置
│   └── restful/                # RESTful API 服务
│
└── tests/                      # 测试代码
```

---

## 4. 核心模块详解

### 4.1 抽象接口层 (backend/clients/api.py)

这是整个项目的**核心抽象**，所有数据库客户端必须实现 `VectorDB` 类：

```python
class VectorDB(ABC):
    """向量数据库抽象基类"""

    @abstractmethod
    def __init__(self, dim, db_config, db_case_config, collection_name, drop_old=False):
        """初始化数据库连接、创建/删除集合"""
        pass

    @abstractmethod
    @contextmanager
    def init(self):
        """创建数据库连接的上下文管理器（多进程安全）"""
        pass

    @abstractmethod
    def insert_embeddings(self, embeddings, metadata, labels_data=None):
        """插入向量数据，返回 (插入数量, 异常)"""
        pass

    @abstractmethod
    def search_embedding(self, query, k=100):
        """向量搜索，返回 Top-K 最近邻的 ID 列表"""
        pass

    @abstractmethod
    def optimize(self, data_size=None):
        """优化索引（如构建索引、force merge），在插入后调用"""
        pass
```

**配置类：**
- `DBConfig`：数据库连接配置（host、port、password 等）
- `DBCaseConfig`：用例级配置（HNSW 参数、IVF 参数等）
- `MetricType`：距离度量类型（L2、COSINE、IP）
- `IndexType`：索引类型枚举（HNSW、IVF_FLAT、DISKANN 等）

### 4.2 测试用例层 (backend/cases.py)

**CaseType 枚举**定义了所有支持的测试类型：

| 类型 | 说明 | 示例 |
|------|------|------|
| `CapacityDim128/960` | 容量测试 | 重复插入直到内存满 |
| `Performance768D10M` | 性能测试 | 10M 向量，768 维 |
| `Performance768D10M1P` | 过滤性能测试 | 10M 向量，1% 过滤率 |
| `StreamingPerformanceCase` | 流式测试 | 边插入边搜索 |
| `LabelFilterPerformanceCase` | 标签过滤测试 | 字符串标签过滤 |
| `PerformanceCustomDataset` | 自定义数据集 | 使用本地数据集 |

**Case 类继承链：**
```
Case (基类)
├── CapacityCase (容量测试基类)
│   ├── CapacityDim128 (SIFT 500K, 128维)
│   └── CapacityDim960 (GIST 100K, 960维)
├── PerformanceCase (性能测试基类)
│   ├── Performance768D100M (LAION 100M)
│   ├── Performance768D10M (Cohere 10M)
│   ├── Performance768D1M (Cohere 1M)
│   ├── Performance1536D5M (OpenAI 5M)
│   ├── IntFilterPerformanceCase (整数过滤)
│   └── ...
└── StreamingPerformanceCase (流式测试)
```

### 4.3 数据集管理 (backend/dataset.py)

**内置数据集：**

| 数据集 | 维度 | 度量类型 | 支持规模 | 用途 |
|--------|------|---------|---------|------|
| SIFT | 128 | L2 | 500K, 5M | 小维度基准 |
| GIST | 960 | L2 | 100K, 1M | 大维度基准 |
| Cohere | 768 | COSINE | 100K, 1M, 10M | 中维度基准 |
| OpenAI | 1536 | COSINE | 50K, 500K, 5M | Embedding 基准 |
| LAION | 768 | L2 | 100M | 超大规模测试 |
| Bioasq | 1024 | COSINE | 1M, 10M | 医疗领域 |

**DatasetManager** 职责：
1. 从 S3/AliyunOSS 下载数据集到本地 `/tmp/vectordb_bench/dataset/`
2. 管理训练数据 (`train.parquet`)、测试数据 (`test.parquet`)、 Ground Truth (`neighbors.parquet`)
3. 提供迭代器接口，分批读取数据
4. 支持自定义本地数据集

### 4.4 任务执行流程 (backend/task_runner.py)

**TaskRunner** 管理整个测试任务的生命周期：

```
TaskRunner (一个任务包含多个 Case)
    └── CaseRunner[] (每个 Case 对应一个数据库 + 一个测试场景)
        ├── init_db()          # 初始化数据库连接
        ├── _load_train_data() # 并发插入训练数据
        ├── _optimize()        # 构建索引/优化
        ├── _serial_search()   # 串行搜索（计算 Recall/Latency）
        └── _conc_search()     # 并发搜索（计算 QPS）
```

**执行引擎 (runner/)：**

| Runner | 文件 | 用途 |
|--------|------|------|
| SerialInsertRunner | serial_runner.py | 串行插入（容量测试） |
| ConcurrentInsertRunner | concurrent_runner.py | 并发插入（性能测试） |
| SerialSearchRunner | serial_runner.py | 串行搜索，计算 Recall |
| MultiProcessingSearchRunner | mp_runner.py | 多进程并发搜索，计算 QPS |
| ReadWriteRunner | read_write_runner.py | 读写混合（流式测试） |

### 4.5 结果与指标 (metric.py, models.py)

**Metric 数据类包含的字段：**

```python
@dataclass
class Metric:
    max_load_count: int = 0          # 容量测试：最大加载数量
    insert_duration: float = 0.0     # 插入耗时
    optimize_duration: float = 0.0   # 优化耗时
    load_duration: float = 0.0       # 总加载耗时
    qps: float = 0.0                 # 最大 QPS
    serial_latency_p99: float = 0.0  # P99 延迟
    serial_latency_p95: float = 0.0  # P95 延迟
    recall: float = 0.0              # 召回率
    ndcg: float = 0.0                # NDCG
```

**结果存储：**
- 结果文件：`results/{db_name}/result_{日期}_{标签}_{db}.json`
- 使用 `TestResult.flush()` 自动保存
- 支持通过 `BenchMarkRunner.get_results()` 读取历史结果

---

## 5. 完整生命周期流程

### 5.1 GUI 模式启动流程

```
用户执行 init_bench
    │
    ▼
vectordb_bench.__main__.py:main()
    │
    ▼
run_streamlit()
    │
    ▼
streamlit run frontend/vdbbench.py
    │
    ▼
vdbbench.py:main()
    ├── set_page_config()          # 配置页面标题、布局
    ├── drawHeaderIcon()           # 绘制标题图标
    ├── initStyle()                # 初始化样式
    ├── welcomePrams()             # 欢迎页面
    └── explainPrams()             # 参数说明
```

### 5.2 测试执行完整流程

```
用户在 GUI 选择数据库和测试用例并提交
    │
    ▼
interface.py: BenchMarkRunner.run(tasks, task_label)
    │
    ├── 生成 run_id (UUID)
    ├── 调用 Assembler.assemble_all() 组装 CaseRunner 列表
    └── 启动异步进程执行 _async_task_v2()
            │
            ▼
    对于每个 CaseRunner:
            │
            ├── CaseRunner._pre_run()
            │       ├── init_db()              # 创建数据库连接和集合
            │       └── dataset.prepare()      # 下载数据集到本地
            │
            ├── 如果是 PerformanceCase:
            │       ├── _load_train_data()     # 并发插入数据
            │       ├── _optimize()            # 构建索引（带超时控制）
            │       ├── _init_search_runner()  # 准备测试数据
            │       ├── _serial_search()       # 串行搜索 -> Recall, Latency
            │       └── _conc_search()         # 并发搜索 -> QPS
            │
            ├── 如果是 CapacityCase:
            │       └── SerialInsertRunner.run_endlessness() # 无限插入直到失败
            │
            └── 如果是 StreamingCase:
                    └── ReadWriteRunner.run_read_write() # 边写边读
            │
            ▼
    收集所有 CaseResult -> 组成 TestResult
    TestResult.flush() -> 保存 JSON 结果文件
    send_conn.send(SIGNAL.SUCCESS)
```

### 5.3 CLI 模式执行流程

```
用户执行: vectordbbench pgvectorhnsw --user-name ... --case-type Performance768D10M
    │
    ▼
cli/vectordbbench.py: 找到对应命令 (pgvectorhnsw)
    │
    ▼
cli/cli.py: run(db, db_config, db_case_config, **parameters)
    │
    ├── 解析参数 -> TaskConfig
    │   ├── db=DB.PgVector
    │   ├── db_config=PgVectorConfig(...)
    │   ├── db_case_config=HNSWConfig(...)
    │   ├── case_config=CaseConfig(case_id=Performance768D10M, ...)
    │   └── stages=[DROP_OLD, LOAD, SEARCH_SERIAL, SEARCH_CONCURRENT]
    │
    ├── 调用 benchmark_runner.run([task], task_label)
    │
    └── 轮询 benchmark_runner.has_running() 直到完成
```

---

## 6. 如何开始

### 6.1 环境要求

- **Python**: >= 3.11
- **OS**: Linux/macOS/Windows (推荐 Linux)
- **内存**: 根据测试规模，建议 >= 32GB（大用例需要更多）
- **磁盘**: 数据集缓存需要大量空间（100M 用例约 300GB+）

### 6.2 安装

**基础安装（仅 Milvus 客户端）：**
```bash
pip install vectordb-bench
```

**安装特定数据库客户端：**
```bash
# PgVector
pip install 'vectordb-bench[pgvector]'

# Pinecone
pip install 'vectordb-bench[pinecone]'

# Elasticsearch
pip install 'vectordb-bench[elastic]'

# 多个客户端
pip install 'vectordb-bench[pgvector,pinecone,qdrant]'
```

**开发安装（从源码）：**
```bash
git clone https://github.com/zilliztech/VectorDBBench.git
cd VectorDBBench
pip install -e '.[test]'
pip install -e '.[pgvector]'  # 安装你需要的数据库客户端
```

### 6.3 启动 GUI

```bash
# 方式 1: 使用命令
init_bench

# 方式 2: 使用 Python 模块
python -m vectordb_bench

# 方式 3: 直接启动 Streamlit
streamlit run vectordb_bench/frontend/vdbbench.py
```

启动后访问 `http://localhost:8501`

### 6.4 环境变量配置

| 环境变量 | 默认值 | 说明 |
|---------|--------|------|
| `LOG_LEVEL` | INFO | 日志级别 |
| `LOG_FILE` | logs/vectordb_bench.log | 日志文件路径 |
| `DATASET_SOURCE` | S3 | 数据集来源 (S3/AliyunOSS) |
| `DATASET_LOCAL_DIR` | /tmp/vectordb_bench/dataset | 数据集本地缓存目录 |
| `RESULTS_LOCAL_DIR` | vectordb_bench/results | 结果存储目录 |
| `CONFIG_LOCAL_DIR` | vectordb_bench/config-files | 配置文件目录 |
| `NUM_PER_BATCH` | 100 | 每批插入向量数 |
| `NUM_CONCURRENCY` | 1,5,10,20,30,40,60,80 | 并发测试的并发数列表 |
| `CONCURRENCY_DURATION` | 30 | 每个并发级别测试时长(秒) |
| `DROP_OLD` | True | 是否删除旧数据 |

---

## 7. 常见使用方式

### 7.1 GUI 模式使用

**运行测试：**
1. 打开 `http://localhost:8501`
2. 选择 **Run Test** 页面
3. 选择要测试的数据库（可多选）
4. 填写数据库连接信息（host、port、user、password）
5. 选择测试用例（如 Performance768D10M）
6. 填写任务标签（用于区分不同测试结果）
7. 点击 Submit 开始测试

**查看结果：**
1. 选择 **Results** 页面查看历史结果
2. 可以对比多个数据库的性能
3. 查看 QPS-Recall 曲线

**自定义数据集：**
1. 选择 **Custom** 页面
2. 上传本地数据集（需符合 Parquet 格式要求）
3. 在 Run Test 页面选择自定义用例

### 7.2 CLI 模式使用

**基本命令结构：**
```bash
vectordbbench [数据库命令] [选项]
```

**查看支持的数据库：**
```bash
vectordbbench --help
```

**运行 PgVector HNSW 测试：**
```bash
vectordbbench pgvectorhnsw \
  --user-name postgres \
  --password 'your_password' \
  --host localhost \
  --db-name vectordb \
  --case-type Performance768D10M \
  --m 16 \
  --ef-construction 128 \
  --ef-search 128 \
  --task-label "my-test"
```

**运行 Milvus 测试：**
```bash
vectordbbench milvushnsw \
  --uri http://localhost:19530 \
  --case-type Performance1536D50K \
  --m 16 \
  --ef-construction 128 \
  --ef-search 128
```

**只执行搜索（跳过加载）：**
```bash
vectordbbench pgvectorhnsw \
  --host localhost \
  --user-name postgres \
  --password 'pass' \
  --case-type Performance768D1M \
  --skip-load \
  --skip-drop-old
```

**自定义并发数：**
```bash
vectordbbench pgvectorhnsw \
  --host localhost \
  --user-name postgres \
  --password 'pass' \
  --case-type Performance768D1M \
  --num-concurrency 1,10,20,50,100 \
  --concurrency-duration 60
```

**使用配置文件：**
```bash
# 创建 config.yaml
vectordbbench pgvectorhnsw --config-file my-config.yaml
```

**批量执行：**
```bash
# 创建 batch-config.yaml，包含多个测试配置
vectordbbench batchcli --batch-config-file batch-config.yaml
```

### 7.3 RESTful API 模式

```bash
# 启动 RESTful 服务
init_bench_rest

# 然后通过 HTTP API 提交测试任务
```

### 7.4 结果分析

**查看结果文件：**
```bash
ls vectordb_bench/results/
# 结果按数据库分类存储
# 例如: results/pgvector/result_20240115_my-test_pgvector.json
```

**结果文件结构：**
```json
{
  "run_id": "abc123",
  "task_label": "my-test",
  "timestamp": 1705312800,
  "results": [
    {
      "metrics": {
        "load_duration": 120.5,
        "qps": 1500.0,
        "recall": 0.98,
        "serial_latency_p99": 5.2
      },
      "task_config": {
        "db": "PgVector",
        "case_config": {"case_id": "Performance768D10M"}
      },
      "label": ":)"
    }
  ]
}
```

---

## 8. 如何添加新的数据库客户端

添加一个新数据库只需 **4 个步骤**：

### Step 1: 创建客户端目录和文件

在 `vectordb_bench/backend/clients/` 下创建 `mydb/` 目录：
```
mydb/
├── __init__.py
├── config.py       # 数据库配置
└── mydb.py         # 客户端实现
```

### Step 2: 实现 config.py

```python
from pydantic import SecretStr
from vectordb_bench.backend.clients.api import DBConfig, DBCaseConfig

class MyDBConfig(DBConfig):
    """连接配置"""
    host: str
    port: int = 8080
    user: str
    password: SecretStr

    def to_dict(self) -> dict:
        return {
            "host": self.host,
            "port": self.port,
            "user": self.user,
            "password": self.password.get_secret_value(),
        }

class MyDBCaseConfig(DBCaseConfig):
    """用例配置（如索引参数）"""
    index_type: str = "HNSW"
    m: int = 16
    ef_construction: int = 128

    def index_param(self) -> dict:
        return {"index_type": self.index_type, "m": self.m}

    def search_param(self) -> dict:
        return {"ef_search": self.ef_search}
```

### Step 3: 实现 mydb.py

```python
from contextlib import contextmanager
from vectordb_bench.backend.clients.api import VectorDB

class MyDB(VectorDB):
    name = "MyDB"

    def __init__(self, dim, db_config, db_case_config, collection_name, drop_old=False):
        self.dim = dim
        self.db_config = db_config
        self.case_config = db_case_config
        self.collection_name = collection_name
        # 初始化客户端...

    @contextmanager
    def init(self):
        # 建立连接
        self.client = MyDBClient(**self.db_config)
        try:
            yield
        finally:
            self.client.close()

    def insert_embeddings(self, embeddings, metadata, labels_data=None, **kwargs):
        # 插入数据
        self.client.insert(embeddings, metadata)
        return len(embeddings), None

    def search_embedding(self, query, k=100):
        # 执行搜索
        results = self.client.search(query, top_k=k)
        return [r.id for r in results]

    def optimize(self, data_size=None):
        # 构建索引/优化
        self.client.build_index()
```

### Step 4: 注册到系统

编辑 `vectordb_bench/backend/clients/__init__.py`：

```python
from enum import Enum

class DB(Enum):
    # ... 其他数据库
    MyDB = "MyDB"

    @property
    def init_cls(self):
        if self == DB.MyDB:
            from .mydb.mydb import MyDB
            return MyDB
        # ...

    @property
    def config_cls(self):
        if self == DB.MyDB:
            from .mydb.config import MyDBConfig
            return MyDBConfig
        # ...

    def case_config_cls(self, index_type=None):
        if self == DB.MyDB:
            from .mydb.config import MyDBCaseConfig
            return MyDBCaseConfig
        # ...
```

**可选 Step 5: 添加 CLI 支持**

创建 `vectordb_bench/backend/clients/mydb/cli.py`：
```python
from typing import Annotated, Unpack
import click
from vectordb_bench.cli.cli import cli, CommonTypedDict, click_parameter_decorators_from_typed_dict, run
from vectordb_bench.backend.clients import DB

class MyDBTypedDict(CommonTypedDict):
    host: Annotated[str, click.option("--host", type=str, required=True)]
    port: Annotated[int, click.option("--port", type=int, default=8080)]
    user: Annotated[str, click.option("--user", type=str, required=True)]
    password: Annotated[str, click.option("--password", type=str, required=True)]

@cli.command()
@click_parameter_decorators_from_typed_dict(MyDBTypedDict)
def mydb(**parameters: Unpack[MyDBTypedDict]):
    from .config import MyDBConfig, MyDBCaseConfig
    run(
        db=DB.MyDB,
        db_config=MyDBConfig(...),
        db_case_config=MyDBCaseConfig(...),
        **parameters,
    )
```

然后在 `vectordb_bench/cli/vectordbbench.py` 中导入：
```python
from vectordb_bench.backend.clients.mydb import cli as mydb_cli
```

---

## 9. 核心设计模式总结

### 9.1 模板方法模式
- `Case` 基类定义测试流程，`CapacityCase`、`PerformanceCase`、`StreamingPerformanceCase` 实现具体逻辑
- `VectorDB` 抽象基类定义数据库接口，各客户端实现具体方法

### 9.2 策略模式
- `Filter` 基类支持多种过滤策略：`NonFilter`、`IntFilter`、`LabelFilter`
- `DatasetSource` 支持多种数据源：`S3`、`AliyunOSS`

### 9.3 工厂模式
- `DB` 枚举类根据名称返回对应的客户端类、配置类
- `CaseType` 枚举根据类型创建对应的 Case 实例

### 9.4 观察者模式
- `BenchMarkRunner` 使用 `multiprocessing.Pipe` 进行进程间通信
- `SIGNAL` 枚举定义任务状态：SUCCESS、ERROR、WIP

### 9.5 建造者模式
- `Assembler` 将 `TaskConfig` 组装成 `CaseRunner`
- `DatasetManager` 负责准备和加载数据集

---

## 10. 常见问题与最佳实践

### 10.1 数据集下载失败
- 检查网络连接，确认能访问 S3 或 AliyunOSS
- 设置 `DATASET_LOCAL_DIR` 指向有足够空间的目录
- 对于大用例（100M），建议提前下载数据集

### 10.2 内存不足
- 减少 `NUM_PER_BATCH`（默认 100）
- 减少并发数 `NUM_CONCURRENCY`
- 小批量测试先用 50K/500K 用例验证

### 10.3 添加数据库的最佳实践
- 先实现最简版本（只支持基础搜索）
- 使用 `thread_safe = True/False` 声明线程安全性
- 实现 `need_normalize_cosine()` 如果数据库不支持 COSINE 原生距离
- 支持 `supported_filter_types` 声明支持的过滤类型

### 10.4 调试技巧
```bash
# 开启 DEBUG 日志
LOG_LEVEL=DEBUG init_bench

# 只运行加载阶段（跳过搜索）
vectordbbench pgvectorhnsw ... --skip-search-serial --skip-search-concurrent

# Dry-run 查看配置是否正确
vectordbbench pgvectorhnsw ... --dry-run
```

---

## 11. 代码规范

项目使用 `black` 和 `ruff` 进行代码格式化：

```bash
# 检查代码风格
make lint

# 自动修复格式
make format
```

- 行长度：120 字符
- Python 版本：3.11+
- 类型注解：推荐使用

---

*文档生成时间: 2024年*  
*基于 VectorDBBench 代码库自动生成*
