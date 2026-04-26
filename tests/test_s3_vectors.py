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
