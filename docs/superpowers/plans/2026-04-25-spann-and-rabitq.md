# SPANN and SPANN_RABITQ Index Support Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add SPANN and SPANN_RABITQ as new Milvus index types with zero default parameters (params dict is empty unless explicitly configured).

**Architecture:** Follow the existing Milvus index configuration pattern: add entries to `IndexType` enum, create pydantic config classes inheriting `MilvusIndexConfig` + `DBCaseConfig`, register them in the `_milvus_case_config` mapping, wire CLI commands, and expose them in the frontend UI config.

**Tech Stack:** Python 3.11+, pydantic v2, click, pymilvus, streamlit

---

## File Structure

| File | Action | Responsibility |
|------|--------|----------------|
| `vectordb_bench/backend/clients/api.py` | Modify | Add `SPANN` and `SPANN_RABITQ` to `IndexType` StrEnum |
| `vectordb_bench/backend/clients/milvus/config.py` | Modify | Add `SPANNConfig` and `SPANNRABITQConfig` classes; register in `_milvus_case_config` |
| `vectordb_bench/backend/clients/milvus/cli.py` | Modify | Add CLI commands `MilvusSPANN` and `MilvusSPANNRaBitQ` |
| `vectordb_bench/frontend/config/dbCaseConfigs.py` | Modify | Add index types to `CaseConfigParamInput_IndexType` options |
| `tests/test_milvus_spann_config.py` | Create | Pydantic config unit tests |
| `tests/test_milvus_spann_cli.py` | Create | CLI registration tests |
| `tests/test_milvus_spann_frontend.py` | Create | Frontend dropdown tests |
| `tests/test_milvus_spann_e2e.py` | Create | End-to-end resolution + regression tests |

**Naming convention:** Class names follow the existing pattern in `milvus/config.py` — drop internal underscores from enum names (e.g., enum `IVF_RABITQ` → class `IVFRABITQConfig`). Therefore:
- `IndexType.SPANN` → class `SPANNConfig`
- `IndexType.SPANN_RABITQ` → class `SPANNRABITQConfig` (no underscore between SPANN and RABITQ)

---

### Task 1: Add SPANN and SPANN_RABITQ to IndexType enum

**Files:**
- Modify: `vectordb_bench/backend/clients/api.py:20-54`

- [ ] **Step 1: Add enum values**

In `vectordb_bench/backend/clients/api.py`, insert two new lines after `IVF_RABITQ = "IVF_RABITQ"` (currently line 32). The full edited enum should look like:

```python
class IndexType(StrEnum):
    HNSW = "HNSW"
    HNSW_SQ = "HNSW_SQ"
    HNSW_BQ = "HNSW_BQ"
    HNSW_PQ = "HNSW_PQ"
    HNSW_PRQ = "HNSW_PRQ"
    DISKANN = "DISKANN"
    STREAMING_DISKANN = "DISKANN"
    IVFFlat = "IVF_FLAT"
    IVFPQ = "IVF_PQ"
    IVFBQ = "IVF_BQ"
    IVFSQ8 = "IVF_SQ8"
    IVF_RABITQ = "IVF_RABITQ"
    SPANN = "SPANN"
    SPANN_RABITQ = "SPANN_RABITQ"
    Flat = "FLAT"
    AUTOINDEX = "AUTOINDEX"
    # ... rest unchanged
```

Use Edit tool with `old_string="    IVF_RABITQ = \"IVF_RABITQ\"\n    Flat = \"FLAT\""` and `new_string="    IVF_RABITQ = \"IVF_RABITQ\"\n    SPANN = \"SPANN\"\n    SPANN_RABITQ = \"SPANN_RABITQ\"\n    Flat = \"FLAT\""`.

- [ ] **Step 2: Verify both enums resolve**

Run:
```bash
python -c "from vectordb_bench.backend.clients.api import IndexType; print(IndexType.SPANN.value, IndexType.SPANN_RABITQ.value)"
```
Expected stdout: `SPANN SPANN_RABITQ`

- [ ] **Step 3: Verify no existing enum values broke**

Run:
```bash
python -c "from vectordb_bench.backend.clients.api import IndexType; assert IndexType.HNSW.value == 'HNSW' and IndexType.IVF_RABITQ.value == 'IVF_RABITQ' and IndexType.AUTOINDEX.value == 'AUTOINDEX'; print('ok')"
```
Expected stdout: `ok`

- [ ] **Step 4: Commit**

```bash
git add vectordb_bench/backend/clients/api.py
git commit -m "feat(milvus): add SPANN and SPANN_RABITQ to IndexType enum"
```

---

### Task 2: Add SPANNConfig and SPANNRABITQConfig classes

**Files:**
- Modify: `vectordb_bench/backend/clients/milvus/config.py`
- Create: `tests/test_milvus_spann_config.py`

- [ ] **Step 1: Write failing test**

Create `tests/test_milvus_spann_config.py`:

```python
from vectordb_bench.backend.clients.api import IndexType, MetricType
from vectordb_bench.backend.clients.milvus.config import (
    SPANNConfig,
    SPANNRABITQConfig,
    _milvus_case_config,
)


def test_spann_index_param():
    cfg = SPANNConfig(metric_type=None)
    assert cfg.index_param() == {"metric_type": "", "index_type": "SPANN", "params": {}}


def test_spann_search_param():
    cfg = SPANNConfig(metric_type=None)
    assert cfg.search_param() == {"metric_type": ""}


def test_spann_rabitq_index_param():
    cfg = SPANNRABITQConfig(metric_type=None)
    assert cfg.index_param() == {"metric_type": "", "index_type": "SPANN_RABITQ", "params": {}}


def test_spann_rabitq_search_param():
    cfg = SPANNRABITQConfig(metric_type=None)
    assert cfg.search_param() == {"metric_type": ""}


def test_spann_with_metric():
    cfg = SPANNConfig(metric_type=MetricType.L2)
    assert cfg.index_param()["metric_type"] == "L2"
    assert cfg.search_param()["metric_type"] == "L2"


def test_spann_rabitq_with_metric():
    cfg = SPANNRABITQConfig(metric_type=MetricType.COSINE)
    assert cfg.index_param()["metric_type"] == "COSINE"
    assert cfg.search_param()["metric_type"] == "COSINE"


def test_spann_params_is_empty_dict():
    """Strict check: params must be {}, not None or missing."""
    cfg = SPANNConfig(metric_type=None)
    idx = cfg.index_param()
    assert "params" in idx
    assert idx["params"] == {}
    assert len(idx["params"]) == 0


def test_spann_rabitq_params_is_empty_dict():
    cfg = SPANNRABITQConfig(metric_type=None)
    idx = cfg.index_param()
    assert "params" in idx
    assert idx["params"] == {}
    assert len(idx["params"]) == 0


def test_spann_no_extra_keys_in_index_param():
    """Ensure no stray keys leak into index params."""
    cfg = SPANNConfig(metric_type=None)
    assert set(cfg.index_param().keys()) == {"metric_type", "index_type", "params"}


def test_spann_no_extra_keys_in_search_param():
    cfg = SPANNConfig(metric_type=None)
    assert set(cfg.search_param().keys()) == {"metric_type"}


def test_spann_rabitq_no_extra_keys_in_index_param():
    cfg = SPANNRABITQConfig(metric_type=None)
    assert set(cfg.index_param().keys()) == {"metric_type", "index_type", "params"}


def test_spann_rabitq_no_extra_keys_in_search_param():
    cfg = SPANNRABITQConfig(metric_type=None)
    assert set(cfg.search_param().keys()) == {"metric_type"}


def test_spann_pydantic_serialization():
    """Verify config can be serialized and deserialized."""
    cfg = SPANNConfig(metric_type=MetricType.IP)
    data = cfg.model_dump()
    assert data["index"] == "SPANN"
    restored = SPANNConfig(**data)
    assert restored.index_param() == cfg.index_param()


def test_spann_rabitq_pydantic_serialization():
    cfg = SPANNRABITQConfig(metric_type=MetricType.IP)
    data = cfg.model_dump()
    assert data["index"] == "SPANN_RABITQ"
    restored = SPANNRABITQConfig(**data)
    assert restored.index_param() == cfg.index_param()


def test_milvus_case_config_mapping():
    assert IndexType.SPANN in _milvus_case_config
    assert IndexType.SPANN_RABITQ in _milvus_case_config
    assert _milvus_case_config[IndexType.SPANN] is SPANNConfig
    assert _milvus_case_config[IndexType.SPANN_RABITQ] is SPANNRABITQConfig
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_milvus_spann_config.py -v`
Expected: FAIL with `ImportError: cannot import name 'SPANNConfig' from 'vectordb_bench.backend.clients.milvus.config'`

- [ ] **Step 3: Add config classes to milvus/config.py**

In `vectordb_bench/backend/clients/milvus/config.py`, insert the two new classes immediately after the `SVSVamanaLeanVecConfig` class (which currently ends at line 492) and immediately before the `_milvus_case_config = {...}` dict (currently at line 495). This keeps class declaration order aligned with the mapping order.

```python
class SPANNConfig(MilvusIndexConfig, DBCaseConfig):
    index: IndexType = IndexType.SPANN

    def index_param(self) -> dict:
        return {
            "metric_type": self.parse_metric(),
            "index_type": self.index.value,
            "params": {},
        }

    def search_param(self) -> dict:
        return {
            "metric_type": self.parse_metric(),
        }


class SPANNRABITQConfig(MilvusIndexConfig, DBCaseConfig):
    index: IndexType = IndexType.SPANN_RABITQ

    def index_param(self) -> dict:
        return {
            "metric_type": self.parse_metric(),
            "index_type": self.index.value,
            "params": {},
        }

    def search_param(self) -> dict:
        return {
            "metric_type": self.parse_metric(),
        }
```

Use Edit tool with `old_string` ending at the closing brace of `SVSVamanaLeanVecConfig.index_param` plus the blank line before `_milvus_case_config`, and append the two new classes.

- [ ] **Step 4: Register in _milvus_case_config mapping**

Append two entries to the existing `_milvus_case_config` dict (currently at line 495 in `vectordb_bench/backend/clients/milvus/config.py`). The full edited dict should look like:

```python
_milvus_case_config = {
    IndexType.AUTOINDEX: AutoIndexConfig,
    IndexType.HNSW: HNSWConfig,
    IndexType.HNSW_SQ: HNSWSQConfig,
    IndexType.HNSW_PQ: HNSWPQConfig,
    IndexType.HNSW_PRQ: HNSWPRQConfig,
    IndexType.DISKANN: DISKANNConfig,
    IndexType.IVFFlat: IVFFlatConfig,
    IndexType.IVFPQ: IVFPQConfig,
    IndexType.IVFSQ8: IVFSQ8Config,
    IndexType.IVF_RABITQ: IVFRABITQConfig,
    IndexType.Flat: FLATConfig,
    IndexType.GPU_IVF_FLAT: GPUIVFFlatConfig,
    IndexType.GPU_IVF_PQ: GPUIVFPQConfig,
    IndexType.GPU_CAGRA: GPUCAGRAConfig,
    IndexType.GPU_BRUTE_FORCE: GPUBruteForceConfig,
    IndexType.SCANN_MILVUS: SCANNConfig,
    IndexType.SVS_VAMANA: SVSVamanaConfig,
    IndexType.SVS_VAMANA_LVQ: SVSVamanaLVQConfig,
    IndexType.SVS_VAMANA_LEANVEC: SVSVamanaLeanVecConfig,
    IndexType.SPANN: SPANNConfig,
    IndexType.SPANN_RABITQ: SPANNRABITQConfig,
}
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `python -m pytest tests/test_milvus_spann_config.py -v`
Expected: 15 tests PASS (test_spann_index_param, test_spann_search_param, test_spann_rabitq_index_param, test_spann_rabitq_search_param, test_spann_with_metric, test_spann_rabitq_with_metric, test_spann_params_is_empty_dict, test_spann_rabitq_params_is_empty_dict, test_spann_no_extra_keys_in_index_param, test_spann_no_extra_keys_in_search_param, test_spann_rabitq_no_extra_keys_in_index_param, test_spann_rabitq_no_extra_keys_in_search_param, test_spann_pydantic_serialization, test_spann_rabitq_pydantic_serialization, test_milvus_case_config_mapping).

- [ ] **Step 6: Commit**

```bash
git add vectordb_bench/backend/clients/milvus/config.py tests/test_milvus_spann_config.py
git commit -m "feat(milvus): add SPANN and SPANNRABITQ config classes"
```

---

### Task 3: Add CLI commands for SPANN and SPANN_RABITQ

**Files:**
- Modify: `vectordb_bench/backend/clients/milvus/cli.py`
- Create: `tests/test_milvus_spann_cli.py`

The CLI module is loaded as a side effect of importing `vectordb_bench.cli.vectordbbench`, which is what registers each `@cli.command()` from db-specific cli files. Tests must trigger this load before introspecting `cli.commands`.

- [ ] **Step 1: Write failing test**

Create `tests/test_milvus_spann_cli.py`:

```python
import vectordb_bench.cli.vectordbbench  # noqa: F401  (registers all DB CLI commands)
from click.testing import CliRunner

from vectordb_bench.cli.cli import cli


def test_milvus_spann_cli_registered():
    """Verify MilvusSPANN command is registered on the cli group."""
    assert "MilvusSPANN" in cli.commands


def test_milvus_spann_rabitq_cli_registered():
    """Verify MilvusSPANNRaBitQ command is registered on the cli group."""
    assert "MilvusSPANNRaBitQ" in cli.commands


def test_milvus_spann_cli_help_runs():
    """Verify --help renders without crashing (parameters wire up cleanly)."""
    runner = CliRunner()
    result = runner.invoke(cli, ["MilvusSPANN", "--help"])
    assert result.exit_code == 0
    assert "--uri" in result.output


def test_milvus_spann_rabitq_cli_help_runs():
    runner = CliRunner()
    result = runner.invoke(cli, ["MilvusSPANNRaBitQ", "--help"])
    assert result.exit_code == 0
    assert "--uri" in result.output
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_milvus_spann_cli.py -v`
Expected: FAIL with `AssertionError: assert 'MilvusSPANN' in {...}` (the command is not yet registered on the `cli` group).

- [ ] **Step 3: Add CLI commands**

Append to `vectordb_bench/backend/clients/milvus/cli.py` (currently 730 lines; add after the closing `)` of `MilvusGPUCAGRA` at line 730):

```python


class MilvusSPANNTypedDict(CommonTypedDict, MilvusTypedDict): ...


@cli.command()
@click_parameter_decorators_from_typed_dict(MilvusSPANNTypedDict)
def MilvusSPANN(**parameters: Unpack[MilvusSPANNTypedDict]):
    from .config import MilvusConfig, SPANNConfig

    run(
        db=DBTYPE,
        db_config=MilvusConfig(
            db_label=parameters["db_label"],
            uri=SecretStr(parameters["uri"]),
            user=parameters["user_name"],
            password=SecretStr(parameters["password"]) if parameters["password"] else None,
            num_shards=int(parameters["num_shards"]),
            replica_number=int(parameters["replica_number"]),
        ),
        db_case_config=SPANNConfig(),
        **parameters,
    )


class MilvusSPANNRaBitQTypedDict(CommonTypedDict, MilvusTypedDict): ...


@cli.command()
@click_parameter_decorators_from_typed_dict(MilvusSPANNRaBitQTypedDict)
def MilvusSPANNRaBitQ(**parameters: Unpack[MilvusSPANNRaBitQTypedDict]):
    from .config import MilvusConfig, SPANNRABITQConfig

    run(
        db=DBTYPE,
        db_config=MilvusConfig(
            db_label=parameters["db_label"],
            uri=SecretStr(parameters["uri"]),
            user=parameters["user_name"],
            password=SecretStr(parameters["password"]) if parameters["password"] else None,
            num_shards=int(parameters["num_shards"]),
            replica_number=int(parameters["replica_number"]),
        ),
        db_case_config=SPANNRABITQConfig(),
        **parameters,
    )
```

- [ ] **Step 4: Verify the new commands are picked up via the project entry point**

Open `vectordb_bench/cli/vectordbbench.py` and confirm it imports `vectordb_bench.backend.clients.milvus.cli` (this is how `@cli.command()` decorators get registered). If it imports the module wholesale, no further wiring is needed. Run:

```bash
python -c "import vectordb_bench.cli.vectordbbench; from vectordb_bench.cli.cli import cli; assert 'MilvusSPANN' in cli.commands and 'MilvusSPANNRaBitQ' in cli.commands; print('ok')"
```
Expected stdout: `ok`. If this fails, add `from vectordb_bench.backend.clients.milvus import cli as _milvus_cli  # noqa: F401` to `vectordb_bench/cli/vectordbbench.py`.

- [ ] **Step 5: Run tests to verify they pass**

Run: `python -m pytest tests/test_milvus_spann_cli.py -v`
Expected: 4 tests PASS.

- [ ] **Step 6: Commit**

```bash
git add vectordb_bench/backend/clients/milvus/cli.py tests/test_milvus_spann_cli.py
git commit -m "feat(milvus): add SPANN and SPANN_RABITQ CLI commands"
```

---

### Task 4: Add frontend UI support for SPANN and SPANN_RABITQ

**Files:**
- Modify: `vectordb_bench/frontend/config/dbCaseConfigs.py`
- Create: `tests/test_milvus_spann_frontend.py`

`MilvusLoadConfig` and `MilvusPerformanceConfig` already include `CaseConfigParamInput_IndexType`, so simply asserting "the IndexType label is in the list" passes before any change. The tests below verify the *options inside* `CaseConfigParamInput_IndexType` actually contain SPANN/SPANN_RABITQ.

- [ ] **Step 1: Write failing test**

Create `tests/test_milvus_spann_frontend.py`:

```python
from vectordb_bench.backend.clients.api import IndexType
from vectordb_bench.frontend.config.dbCaseConfigs import (
    CaseConfigParamInput_IndexType,
    MilvusLoadConfig,
    MilvusPerformanceConfig,
)


def test_spann_in_index_type_options():
    options = CaseConfigParamInput_IndexType.inputConfig["options"]
    assert IndexType.SPANN.value in options
    assert IndexType.SPANN_RABITQ.value in options


def test_existing_index_types_still_present():
    """Regression: do not lose existing dropdown entries when adding SPANN."""
    options = CaseConfigParamInput_IndexType.inputConfig["options"]
    for required in (
        IndexType.HNSW.value,
        IndexType.IVF_RABITQ.value,
        IndexType.DISKANN.value,
        IndexType.AUTOINDEX.value,
        IndexType.Flat.value,
        IndexType.GPU_CAGRA.value,
    ):
        assert required in options, f"missing existing index option: {required}"


def test_spann_options_reachable_via_milvus_configs():
    """Verify SPANN options are reachable from both Load and Performance config groups."""
    for config_group in (MilvusLoadConfig, MilvusPerformanceConfig):
        index_inputs = [c for c in config_group if c is CaseConfigParamInput_IndexType]
        assert len(index_inputs) == 1
        options = index_inputs[0].inputConfig["options"]
        assert IndexType.SPANN.value in options
        assert IndexType.SPANN_RABITQ.value in options
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_milvus_spann_frontend.py -v`
Expected: FAIL on `test_spann_in_index_type_options` with `AssertionError: assert 'SPANN' in [...]`.

- [ ] **Step 3: Add index types to frontend dropdown**

Edit `CaseConfigParamInput_IndexType` in `vectordb_bench/frontend/config/dbCaseConfigs.py` (currently at line 401). Append two entries to the `options` list:

```python
CaseConfigParamInput_IndexType = CaseConfigInput(
    label=CaseConfigParamType.IndexType,
    inputType=InputType.Option,
    inputConfig={
        "options": [
            IndexType.HNSW.value,
            IndexType.HNSW_SQ.value,
            IndexType.HNSW_PQ.value,
            IndexType.HNSW_PRQ.value,
            IndexType.IVFFlat.value,
            IndexType.IVFPQ.value,
            IndexType.IVFSQ8.value,
            IndexType.IVF_RABITQ.value,
            IndexType.SCANN_MILVUS.value,
            IndexType.DISKANN.value,
            IndexType.Flat.value,
            IndexType.AUTOINDEX.value,
            IndexType.GPU_IVF_FLAT.value,
            IndexType.GPU_IVF_PQ.value,
            IndexType.GPU_CAGRA.value,
            IndexType.GPU_BRUTE_FORCE.value,
            IndexType.SPANN.value,
            IndexType.SPANN_RABITQ.value,
        ],
    },
)
```

No additional parameter inputs are needed for SPANN/SPANN_RABITQ since they carry zero parameters. The empty `params` dict is produced by `SPANNConfig.index_param()` regardless of UI selections.

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/test_milvus_spann_frontend.py -v`
Expected: 3 tests PASS.

- [ ] **Step 5: Commit**

```bash
git add vectordb_bench/frontend/config/dbCaseConfigs.py tests/test_milvus_spann_frontend.py
git commit -m "feat(frontend): add SPANN and SPANN_RABITQ to Milvus UI config"
```

---

### Task 5: End-to-end smoke test and regression guard

**Files:**
- Create: `tests/test_milvus_spann_e2e.py`

- [ ] **Step 1: Write smoke + regression tests**

Create `tests/test_milvus_spann_e2e.py`:

```python
from vectordb_bench.backend.clients import DB
from vectordb_bench.backend.clients.api import IndexType, MetricType
from vectordb_bench.backend.clients.milvus.config import (
    AutoIndexConfig,
    DISKANNConfig,
    FLATConfig,
    HNSWConfig,
    IVFRABITQConfig,
    SPANNConfig,
    SPANNRABITQConfig,
)


def test_db_case_config_cls_resolution():
    """Verify DB.case_config_cls resolves SPANN and SPANN_RABITQ correctly."""
    assert DB.Milvus.case_config_cls(IndexType.SPANN) is SPANNConfig
    assert DB.Milvus.case_config_cls(IndexType.SPANN_RABITQ) is SPANNRABITQConfig


def test_existing_index_types_still_resolve():
    """Regression: adding SPANN must not break existing index resolution."""
    assert DB.Milvus.case_config_cls(IndexType.AUTOINDEX) is AutoIndexConfig
    assert DB.Milvus.case_config_cls(IndexType.DISKANN) is DISKANNConfig
    assert DB.Milvus.case_config_cls(IndexType.Flat) is FLATConfig
    assert DB.Milvus.case_config_cls(IndexType.HNSW) is HNSWConfig
    assert DB.Milvus.case_config_cls(IndexType.IVF_RABITQ) is IVFRABITQConfig


def test_spann_index_param_shape():
    """SPANN config emits the exact 3-key shape with empty params."""
    spann = SPANNConfig(metric_type=MetricType.COSINE)
    idx = spann.index_param()
    assert set(idx.keys()) == {"metric_type", "index_type", "params"}
    assert idx["params"] == {}
    assert idx["index_type"] == "SPANN"
    assert idx["metric_type"] == "COSINE"


def test_spann_rabitq_index_param_shape():
    spann_rq = SPANNRABITQConfig(metric_type=MetricType.L2)
    idx = spann_rq.index_param()
    assert set(idx.keys()) == {"metric_type", "index_type", "params"}
    assert idx["params"] == {}
    assert idx["index_type"] == "SPANN_RABITQ"
    assert idx["metric_type"] == "L2"


def test_spann_search_param_shape():
    """search_param emits ONLY metric_type — no params key — matching plan spec."""
    assert SPANNConfig(metric_type=MetricType.IP).search_param() == {"metric_type": "IP"}
    assert SPANNRABITQConfig(metric_type=MetricType.IP).search_param() == {"metric_type": "IP"}


def test_spann_default_metric_is_empty_string():
    """parse_metric returns '' when metric_type is None — confirms inherited base behavior."""
    assert SPANNConfig().index_param()["metric_type"] == ""
    assert SPANNRABITQConfig().index_param()["metric_type"] == ""
```

- [ ] **Step 2: Run test**

Run: `python -m pytest tests/test_milvus_spann_e2e.py -v`
Expected: 6 tests PASS.

- [ ] **Step 3: Run full Milvus-related test suite to catch regressions**

Run: `python -m pytest tests/test_milvus_spann_config.py tests/test_milvus_spann_cli.py tests/test_milvus_spann_frontend.py tests/test_milvus_spann_e2e.py tests/test_milvus.py -v`
Expected: all tests PASS (counts: 15 + 4 + 3 + 6 + N existing in `test_milvus.py`).

- [ ] **Step 4: Commit**

```bash
git add tests/test_milvus_spann_e2e.py
git commit -m "test: add SPANN/SPANN_RABITQ end-to-end and regression tests"
```

---

## Spec Coverage Checklist

| Requirement | Task |
|-------------|------|
| Add SPANN to `IndexType` enum | Task 1 |
| Add SPANN_RABITQ to `IndexType` enum | Task 1 |
| SPANN config class with empty default params | Task 2 |
| SPANN_RABITQ config class with empty default params | Task 2 |
| Register both in `_milvus_case_config` mapping | Task 2 |
| CLI command for SPANN | Task 3 |
| CLI command for SPANN_RABITQ | Task 3 |
| Entry-point wiring for new CLI commands | Task 3 (Step 4) |
| Frontend dropdown includes SPANN | Task 4 |
| Frontend dropdown includes SPANN_RABITQ | Task 4 |
| Frontend dropdown does not lose existing entries | Task 4 (regression test) |
| Zero parameters by default (empty `params` dict) | Tasks 2, 5 |
| `DB.Milvus.case_config_cls()` resolves new types | Task 5 |
| Existing index types still resolve | Task 5 (regression test) |

## Placeholder Scan

- No "TBD", "TODO", or "implement later" present.
- All test code contains concrete assertions.
- All file paths and line numbers are verified against the worktree (commit `88ad032`).
- All commands include expected output.

## Type Consistency Check

- Enum members: `IndexType.SPANN` / `IndexType.SPANN_RABITQ` used consistently.
- Config class names: `SPANNConfig` and `SPANNRABITQConfig` (no internal underscore — matches `IVFRABITQConfig` precedent) used consistently in Tasks 2, 3, 5.
- CLI command names: `MilvusSPANN` and `MilvusSPANNRaBitQ` used consistently in the CLI module and the CLI tests (Task 3).
- TypedDict names: `MilvusSPANNTypedDict` and `MilvusSPANNRaBitQTypedDict` used consistently within Task 3.

## Risks and Mitigations

| Risk | Mitigation |
|------|-----------|
| New CLI command not auto-registered if `vectordb_bench/cli/vectordbbench.py` does not import the milvus cli module | Task 3 Step 4 verifies registration and provides the exact import line to add if needed |
| Empty `params: {}` rejected by pymilvus server-side | Task 2 unit tests assert the dict shape; runtime validation requires a live Milvus instance and is out of scope for this plan |
| Frontend lazy-import of streamlit can mask `dbCaseConfigs.py` import errors during plain `pytest` | Tests in Task 4 import directly from `vectordb_bench.frontend.config.dbCaseConfigs`, bypassing streamlit, so import errors surface immediately |
