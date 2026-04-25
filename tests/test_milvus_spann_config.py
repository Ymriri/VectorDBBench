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
