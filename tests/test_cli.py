"""CLI smoke tests (in-process)."""
from nodeforge import cli


def test_cli_models(capsys):
    assert cli.main(["models"]) == 0
    out = capsys.readouterr().out
    assert "adaprop" in out and "logreg" in out


def test_cli_info(capsys):
    assert cli.main(["info"]) == 0
    assert "seed" in capsys.readouterr().out


def test_cli_run(capsys):
    assert cli.main(["run", "--tier", "tier1_homophilous_noisy",
                     "--model", "logreg", "--seed", "42"]) == 0
    assert "logreg" in capsys.readouterr().out


def test_cli_run_bad_tier(capsys):
    assert cli.main(["run", "--tier", "bad"]) == 1
