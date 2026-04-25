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
