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
