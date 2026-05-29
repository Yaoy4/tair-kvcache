"""Phase 6c — A/B equivalence comparator.

Classifies two ``RunResult`` objects (one per backend) against the
tolerance spec in ``hisim/docs/pd_ab_equivalence_spec.md``.

Failure-mode classes (first match wins):

* ``STRUCT``     — any of invariants 1–6 fails. Hard CI block.
* ``PER_REQ``    — at least one rid exceeds per-request tolerance.
* ``AGGREGATE``  — only aggregate (mean / pXX) tolerances fail.
* ``WARN``       — only allowed-asymmetry items differ. Non-blocking.
* ``OK``         — pairwise identical within all numeric tolerances.

This module imports nothing backend-specific and has no I/O. It is the
single source of truth for "are A and B equivalent?"; the harness in 6b
just produces inputs and the CI gate in 6e calls ``classify(...)``.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import List, Optional, Sequence

from hisim.simulation.pd_ab_harness import RequestResult, RunResult


# ---------------------------------------------------------------------------
# Tolerances (from spec — keep in sync with hisim/docs/pd_ab_equivalence_spec.md).
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Tolerance:
    abs_tol: float = 1e-6
    rel_tol: float = 1e-9

    def within(self, a: float, b: float) -> bool:
        return abs(a - b) <= max(self.abs_tol, self.rel_tol * max(abs(a), abs(b)))


DEFAULT_TOLERANCE = Tolerance(abs_tol=1e-6, rel_tol=1e-9)


# Per-request fields the spec pins (all on the same tight bound).
_PER_REQ_TIMESTAMP_FIELDS = (
    "prefill_start_time",
    "prefill_end_time",
    "kv_ready_time",
    "decode_start_time",
    "decode_end_time",
)
# Derived per-request quantities the spec also pins.
_PER_REQ_DERIVED = (
    "ttft",
    "e2e",
    "prefill_queue_wait",
    "kv_transfer_time",
    "decode_queue_wait",
)


# ---------------------------------------------------------------------------
# Result types.
# ---------------------------------------------------------------------------


class FailureClass(str, Enum):
    OK = "OK"
    STRUCT = "STRUCT"
    PER_REQ = "PER_REQ"
    AGGREGATE = "AGGREGATE"
    WARN = "WARN"


@dataclass
class Mismatch:
    """One concrete A↔B disagreement."""

    kind: FailureClass
    where: str        # e.g. "rid=w1-0.prefill_end_time" or "aggregate.mean_ttft"
    a_value: object
    b_value: object
    detail: str = ""

    def __str__(self) -> str:  # pragma: no cover — cosmetic only
        return f"[{self.kind.value}] {self.where}: A={self.a_value!r} B={self.b_value!r} {self.detail}".strip()


@dataclass
class ComparisonReport:
    classification: FailureClass
    mismatches: List[Mismatch] = field(default_factory=list)
    aggregate_a: dict = field(default_factory=dict)
    aggregate_b: dict = field(default_factory=dict)

    @property
    def is_blocking(self) -> bool:
        """Per spec: STRUCT / PER_REQ / AGGREGATE block; WARN and OK do not."""
        return self.classification in (
            FailureClass.STRUCT,
            FailureClass.PER_REQ,
            FailureClass.AGGREGATE,
        )

    def first_of(self, kind: FailureClass) -> Optional[Mismatch]:
        for m in self.mismatches:
            if m.kind == kind:
                return m
        return None


# ---------------------------------------------------------------------------
# Aggregate computation.
# ---------------------------------------------------------------------------


def _percentile(values: Sequence[float], pct: float) -> float:
    """Nearest-rank percentile (no interpolation). Matches HiSim's
    convention; the exact formula is irrelevant as long as A and B use the
    same one — which they do because we run them through this function."""
    if not values:
        return 0.0
    s = sorted(values)
    k = max(0, min(len(s) - 1, int(round((pct / 100.0) * (len(s) - 1)))))
    return float(s[k])


def _aggregates(run: RunResult) -> dict:
    """Compute the aggregate metrics the spec pins."""
    finished = [r for r in run.per_request if r.finished]
    ttft_vals = [r.ttft for r in finished if r.ttft is not None]
    e2e_vals = [r.e2e for r in finished if r.e2e is not None]
    # ITL = (e2e - ttft) / max(1, output_length - 1). Avoid div-by-zero on
    # 1-token outputs by capping; same formula on A and B = same answer.
    itl_vals = []
    for r in finished:
        if r.e2e is None or r.ttft is None:
            continue
        denom = max(1, int(r.output_length) - 1)
        itl_vals.append((r.e2e - r.ttft) / denom)

    out = {
        "n_finished": float(len(finished)),
        "mean_ttft": _mean(ttft_vals),
        "p50_ttft": _percentile(ttft_vals, 50.0),
        "p95_ttft": _percentile(ttft_vals, 95.0),
        "p99_ttft": _percentile(ttft_vals, 99.0),
        "mean_e2e": _mean(e2e_vals),
        "p50_e2e": _percentile(e2e_vals, 50.0),
        "p95_e2e": _percentile(e2e_vals, 95.0),
        "p99_e2e": _percentile(e2e_vals, 99.0),
        "mean_itl": _mean(itl_vals),
        "p95_itl": _percentile(itl_vals, 95.0),
        "p99_itl": _percentile(itl_vals, 99.0),
    }
    # Throughput: requests/s and output_tokens/s, against final_clock.
    if run.final_clock > 0.0:
        out["throughput_rps"] = len(finished) / run.final_clock
        out["throughput_tok_s"] = sum(r.output_length for r in finished) / run.final_clock
    else:
        out["throughput_rps"] = 0.0
        out["throughput_tok_s"] = 0.0
    return out


def _mean(values: Sequence[float]) -> float:
    return float(sum(values) / len(values)) if values else 0.0


# ---------------------------------------------------------------------------
# Public API: classify(...).
# ---------------------------------------------------------------------------


def classify(
    run_a: RunResult,
    run_b: RunResult,
    *,
    tolerance: Tolerance = DEFAULT_TOLERANCE,
) -> ComparisonReport:
    """Compare ``run_a`` and ``run_b`` and return a classified report.

    The function is total: it always returns a ``ComparisonReport`` and
    never raises on mismatch (it only raises on programmer errors like
    ``None`` inputs).
    """
    if run_a is None or run_b is None:
        raise ValueError("classify() requires two non-None RunResult instances")

    mismatches: List[Mismatch] = []

    # ---- STRUCT invariants (spec §Invariants 1–6) ----
    struct = _check_struct(run_a, run_b)
    mismatches.extend(struct)

    # If STRUCT already failed, downstream per-rid comparison would alarm on
    # the same divergence with less actionable messages. Short-circuit.
    if struct:
        return ComparisonReport(
            classification=FailureClass.STRUCT,
            mismatches=mismatches,
            aggregate_a=_aggregates(run_a),
            aggregate_b=_aggregates(run_b),
        )

    # ---- PER_REQ tolerances ----
    per_req = _check_per_request(run_a, run_b, tolerance)
    mismatches.extend(per_req)

    # ---- AGGREGATE tolerances ----
    agg_a = _aggregates(run_a)
    agg_b = _aggregates(run_b)
    aggregate = _check_aggregate(agg_a, agg_b, tolerance)
    mismatches.extend(aggregate)

    # ---- classify (first failure wins per spec) ----
    if per_req:
        cls = FailureClass.PER_REQ
    elif aggregate:
        cls = FailureClass.AGGREGATE
    else:
        cls = FailureClass.OK

    return ComparisonReport(
        classification=cls,
        mismatches=mismatches,
        aggregate_a=agg_a,
        aggregate_b=agg_b,
    )


# ---------------------------------------------------------------------------
# Invariant checks (helpers).
# ---------------------------------------------------------------------------


def _check_struct(run_a: RunResult, run_b: RunResult) -> List[Mismatch]:
    out: List[Mismatch] = []

    # 1. Request-count parity.
    if len(run_a.per_request) != len(run_b.per_request):
        out.append(Mismatch(
            FailureClass.STRUCT, "request_count",
            len(run_a.per_request), len(run_b.per_request),
            "FINISHED request-count parity (invariant 1) failed.",
        ))

    # 2. RID set parity (over FINISHED only — the spec says "FINISHED set").
    a_rids = {r.rid for r in run_a.per_request if r.finished}
    b_rids = {r.rid for r in run_b.per_request if r.finished}
    if a_rids != b_rids:
        out.append(Mismatch(
            FailureClass.STRUCT, "finished_rid_set",
            sorted(a_rids - b_rids), sorted(b_rids - a_rids),
            "FINISHED rid-set parity (invariant 2) failed.",
        ))

    # 3. Output-length parity per rid.
    a_by = run_a.by_rid()
    b_by = run_b.by_rid()
    common = set(a_by) & set(b_by)
    for rid in sorted(common):
        if a_by[rid].output_length != b_by[rid].output_length:
            out.append(Mismatch(
                FailureClass.STRUCT, f"rid={rid}.output_length",
                a_by[rid].output_length, b_by[rid].output_length,
                "Output-length parity (invariant 3) failed.",
            ))

    # 4. Phase progression: monotonic non-decreasing on the canonical sequence.
    for label, run in (("A", run_a), ("B", run_b)):
        for r in run.per_request:
            if not r.finished:
                continue
            seq = [
                ("arrival_time", r.arrival_time),
                ("prefill_start_time", r.prefill_start_time),
                ("prefill_end_time", r.prefill_end_time),
                ("kv_ready_time", r.kv_ready_time),
                ("decode_start_time", r.decode_start_time),
                ("decode_end_time", r.decode_end_time),
            ]
            for (n1, v1), (n2, v2) in zip(seq, seq[1:]):
                if v1 is None or v2 is None:
                    out.append(Mismatch(
                        FailureClass.STRUCT,
                        f"backend={label}.rid={r.rid}.phase_progression",
                        f"{n1}={v1}", f"{n2}={v2}",
                        "FINISHED request with missing timestamp (invariant 4).",
                    ))
                    break
                if v2 < v1:
                    out.append(Mismatch(
                        FailureClass.STRUCT,
                        f"backend={label}.rid={r.rid}.phase_progression",
                        f"{n1}={v1}", f"{n2}={v2}",
                        "Non-monotonic timestamps (invariant 4) failed.",
                    ))
                    break

    # 5. Non-negative stage durations.
    for label, run in (("A", run_a), ("B", run_b)):
        for r in run.per_request:
            for name, val in (
                ("prefill_queue_wait", r.prefill_queue_wait),
                ("kv_transfer_time", r.kv_transfer_time),
                ("decode_queue_wait", r.decode_queue_wait),
            ):
                if val is not None and val < 0.0:
                    out.append(Mismatch(
                        FailureClass.STRUCT,
                        f"backend={label}.rid={r.rid}.{name}",
                        val, ">=0",
                        "Negative stage duration (invariant 5) failed.",
                    ))

    # 6. Schema parity (aggregate key set). Always identical here because
    # _aggregates() is deterministic, but a future caller could pass
    # external dicts — we keep the check for completeness.
    a_keys = set(_aggregates(run_a))
    b_keys = set(_aggregates(run_b))
    if a_keys != b_keys:
        out.append(Mismatch(
            FailureClass.STRUCT, "aggregate_schema",
            sorted(a_keys - b_keys), sorted(b_keys - a_keys),
            "calc_metrics schema parity (invariant 6) failed.",
        ))

    return out


def _check_per_request(
    run_a: RunResult, run_b: RunResult, tol: Tolerance
) -> List[Mismatch]:
    out: List[Mismatch] = []
    a_by = run_a.by_rid()
    b_by = run_b.by_rid()
    for rid in sorted(set(a_by) & set(b_by)):
        ra = a_by[rid]
        rb = b_by[rid]
        for field_name in _PER_REQ_TIMESTAMP_FIELDS + _PER_REQ_DERIVED:
            a_val = getattr(ra, field_name)
            b_val = getattr(rb, field_name)
            if a_val is None and b_val is None:
                continue
            if a_val is None or b_val is None:
                out.append(Mismatch(
                    FailureClass.PER_REQ,
                    f"rid={rid}.{field_name}",
                    a_val, b_val,
                    "One backend reached this stage, the other didn't.",
                ))
                continue
            if not tol.within(a_val, b_val):
                out.append(Mismatch(
                    FailureClass.PER_REQ,
                    f"rid={rid}.{field_name}",
                    a_val, b_val,
                    f"|A-B|={abs(a_val - b_val):.3e} exceeds abs_tol={tol.abs_tol:.0e}.",
                ))
    return out


_AGG_FIELDS = (
    "mean_ttft", "p50_ttft", "p95_ttft", "p99_ttft",
    "mean_e2e",  "p50_e2e",  "p95_e2e",  "p99_e2e",
    "mean_itl",  "p95_itl",  "p99_itl",
    "throughput_rps", "throughput_tok_s",
)


def _check_aggregate(agg_a: dict, agg_b: dict, tol: Tolerance) -> List[Mismatch]:
    out: List[Mismatch] = []
    for k in _AGG_FIELDS:
        if k not in agg_a or k not in agg_b:
            continue  # schema mismatch already reported by STRUCT
        a_val, b_val = float(agg_a[k]), float(agg_b[k])
        if not tol.within(a_val, b_val):
            out.append(Mismatch(
                FailureClass.AGGREGATE,
                f"aggregate.{k}",
                a_val, b_val,
                f"|A-B|={abs(a_val - b_val):.3e} exceeds abs_tol={tol.abs_tol:.0e}.",
            ))
    return out
