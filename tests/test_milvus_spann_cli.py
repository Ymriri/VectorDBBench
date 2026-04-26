import vectordb_bench.cli.vectordbbench  # noqa: F401  (registers all DB CLI commands)
from click.testing import CliRunner

from vectordb_bench.cli.cli import cli


def test_milvus_spann_cli_registered():
    """Verify MilvusSPANN command is registered on the cli group."""
    assert "milvusspann" in cli.commands


def test_milvus_spann_rabitq_cli_registered():
    """Verify MilvusSPANNRaBitQ command is registered on the cli group."""
    assert "milvusspannrabitq" in cli.commands


def test_milvus_spann_cli_help_runs():
    """Verify --help renders without crashing (parameters wire up cleanly)."""
    runner = CliRunner()
    result = runner.invoke(cli, ["milvusspann", "--help"])
    assert result.exit_code == 0
    assert "--uri" in result.output


def test_milvus_spann_rabitq_cli_help_runs():
    runner = CliRunner()
    result = runner.invoke(cli, ["milvusspannrabitq", "--help"])
    assert result.exit_code == 0
    assert "--uri" in result.output
