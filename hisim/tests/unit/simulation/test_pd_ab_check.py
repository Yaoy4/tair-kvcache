"""Phase 6e — CLI gate tests for pd_ab_check."""
from __future__ import annotations

import sys

import pytest

from hisim.simulation import pd_ab_check
from hisim.simulation.pd_ab_check import (
    WorkloadOutcome,
    default_workloads,
    main,
    run_check,
)
from hisim.simulation.pd_ab_compare import (
    ComparisonReport,
    FailureClass,
    Mismatch,
)
from hisim.simulation.pd_ab_harness import RunResult


# ---------------------------------------------------------------------------
# default_workloads sanity.
# ---------------------------------------------------------------------------


def test_default_workloads_includes_w1_through_w6():
    names = [w.name for w in default_workloads()]
    assert names == [
        "W1-single", "W2-serial", "W3-concurrent",
        "W4-long-decode", "W5-long-prefill", "W6-kv-bw-stress",
    ]


# ---------------------------------------------------------------------------
# run_check: smoke on the single cheapest workload (W1) — proves end-to-end
# wiring without burning ~17s on the full matrix.
# ---------------------------------------------------------------------------


def test_run_check_w1_only_returns_ok():
    w1 = [w for w in default_workloads() if w.name == "W1-single"]
    outcomes, blocking = run_check(w1)
    assert len(outcomes) == 1
    assert outcomes[0].name == "W1-single"
    assert outcomes[0].report.classification is FailureClass.OK
    assert outcomes[0].runtime_seconds > 0.0
    assert blocking is False


# ---------------------------------------------------------------------------
# main() exit codes via monkeypatched run_check (fast; no backend builds).
# ---------------------------------------------------------------------------


def _fake_outcome(name: str, cls: FailureClass) -> WorkloadOutcome:
    mismatches = []
    if cls is not FailureClass.OK:
        mismatches = [Mismatch(cls, f"rid=fake.{name}", 1.0, 2.0, "synthetic")]
    return WorkloadOutcome(
        name=name,
        report=ComparisonReport(
            classification=cls,
            mismatches=mismatches,
            aggregate_a={"n_finished": 1.0},
            aggregate_b={"n_finished": 1.0},
        ),
        runtime_seconds=0.01,
    )


def test_main_exit_zero_when_all_workloads_ok(monkeypatch, capsys):
    def fake_run_check(workloads, **_kwargs):
        return [_fake_outcome(w.name, FailureClass.OK) for w in workloads], False

    monkeypatch.setattr(pd_ab_check, "run_check", fake_run_check)
    rc = main(["--workload", "W1-single"])
    assert rc == 0
    out = capsys.readouterr().out
    assert "OK" in out
    assert "0 blocking failure(s)" in out


def test_main_exit_one_on_blocking_failure(monkeypatch, capsys):
    def fake_run_check(workloads, **_kwargs):
        outs = [
            _fake_outcome("W1-single", FailureClass.OK),
            _fake_outcome("W3-concurrent", FailureClass.PER_REQ),
        ]
        return outs, True

    monkeypatch.setattr(pd_ab_check, "run_check", fake_run_check)
    rc = main([])
    assert rc == 1
    out = capsys.readouterr().out
    assert "PER_REQ" in out
    assert "1 blocking failure(s)" in out
    # Verbose-on-fail: the blocking entry must dump its first mismatch.
    assert "rid=fake.W3-concurrent" in out


def test_main_exit_two_on_unknown_workload(capsys):
    rc = main(["--workload", "W99-nope"])
    assert rc == 2
    err = capsys.readouterr().err
    assert "no workloads matched" in err
