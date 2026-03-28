from click.testing import CliRunner
from main import cli

def test_cli_help():
    runner = CliRunner()
    result = runner.invoke(cli, ["--help"])
    assert result.exit_code == 0
    assert "optimize" in result.output or "backtest" in result.output

def test_backtest_missing_args():
    runner = CliRunner()
    result = runner.invoke(cli, ["backtest"])
    assert result.exit_code != 0 or "Error" in result.output or "Missing" in result.output

def test_report_no_runs(tmp_path):
    runner = CliRunner()
    result = runner.invoke(cli, ["report", "--top", "5", "--reports-dir", str(tmp_path)])
    assert result.exit_code == 0
