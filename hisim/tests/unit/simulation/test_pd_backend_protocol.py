"""Phase 5c.0 — PDBackendProtocol structural conformance.

These tests pin the contract that ``BackendA`` and ``BackendB`` are
interchangeable for ``pd_runtime`` helpers and (in 5c.1) for the SGLang hook.

Why this matters: 5c.1 will make ``sglang_hook.wrapped_init`` build A or B
based on ``disagg.backend``. If either backend ever drifts away from the
shared surface (a renamed method, a new required arg), this test catches it
*before* the hook tries to route real traffic through the missing method.
"""
from __future__ import annotations

import inspect

import pytest

from hisim.simulation.pd_backend_a import BackendA
from hisim.simulation.pd_backend_b import BackendB
from hisim.simulation.pd_backend_protocol import PDBackendProtocol
from hisim.simulation.pd_factory import DisaggPredictors
from hisim.simulation.pd_transfer import BandwidthTransferModel, KVModelConfig


# Methods that MUST exist on a backend. Mirrors PDBackendProtocol exactly —
# keeping it explicit catches "I added a method to the protocol but not the
# backend" and vice versa, which a plain Protocol check would silently let
# slide for methods only declared on the protocol.
_REQUIRED_METHODS = (
    "controller",
    "prefill_pool_size",
    "decode_pool_size",
    "earliest_pool_time",
    "try_admit_prefill",
    "compute_kv_ready_time",
    "on_prefill_done",
    "advance_to_kv_ready",
    "try_admit_decode_step",
    "try_admit_decode_batch",
    "on_decode_step_done",
    "on_decode_step_done_batch",
)


def _empty_bundle() -> DisaggPredictors:
    return DisaggPredictors(
        prefill=None,
        decode=None,
        kv_bytes_per_token=1024,
        kv_model_config=KVModelConfig(kv_bytes_per_token=1024),
        transfer_model=BandwidthTransferModel(bw_gbps=100.0, latency_us=10.0),
        prefill_replicas=1,
        decode_replicas=1,
    )


def _StubFactory():  # picklable top-level name not required; never spawned.
    return None


def _make_backend_a() -> BackendA:
    return BackendA(_empty_bundle())


# Note: we never call BackendB.start() — that would spawn workers. The
# protocol check only needs the unbound methods to exist.
def _make_backend_b() -> BackendB:
    return BackendB(
        bundle=_empty_bundle(),
        prefill_predictor_factory=_StubFactory,
        decode_predictor_factory=_StubFactory,
    )


@pytest.mark.parametrize(
    "backend",
    [_make_backend_a(), _make_backend_b()],
    ids=["backend_a", "backend_b"],
)
def test_backend_satisfies_runtime_protocol(backend):
    """runtime_checkable isinstance() must accept both backends."""
    assert isinstance(backend, PDBackendProtocol)


@pytest.mark.parametrize(
    "backend_cls",
    [BackendA, BackendB],
    ids=["backend_a", "backend_b"],
)
def test_backend_class_exposes_all_required_methods(backend_cls):
    for name in _REQUIRED_METHODS:
        method = getattr(backend_cls, name, None)
        assert method is not None, f"{backend_cls.__name__} missing {name}"
        assert callable(method), f"{backend_cls.__name__}.{name} not callable"


@pytest.mark.parametrize(
    "backend_cls",
    [BackendA, BackendB],
    ids=["backend_a", "backend_b"],
)
def test_backend_method_signatures_match_protocol(backend_cls):
    """Argument NAMES must agree between protocol and concrete backend.

    Catches a class of refactor bug where one backend silently renames a
    keyword (e.g. ``req`` → ``request``) — call sites in pd_runtime use
    positional args so it would compile, but kwargs in tests would break
    only for that backend.
    """
    for name in _REQUIRED_METHODS:
        proto_sig = inspect.signature(getattr(PDBackendProtocol, name))
        impl_sig = inspect.signature(getattr(backend_cls, name))
        proto_params = [
            p.name for p in proto_sig.parameters.values() if p.name != "self"
        ]
        impl_params = [
            p.name for p in impl_sig.parameters.values() if p.name != "self"
        ]
        assert proto_params == impl_params, (
            f"{backend_cls.__name__}.{name}{impl_sig} does not match "
            f"PDBackendProtocol.{name}{proto_sig}"
        )


def test_pd_runtime_helpers_accept_either_backend_type():
    """pd_runtime helpers must be annotated against the protocol so the
    type-check surface is single-typed. Spot-check the annotation.
    """
    from hisim.simulation import pd_runtime

    for fn_name in (
        "admit_prefill_batch_latency",
        "decode_batch_latency",
        "finalize_prefill_batch",
        "drain_kv_ready_and_admit_decode",
    ):
        fn = getattr(pd_runtime, fn_name)
        ann = inspect.signature(fn).parameters["backend"].annotation
        # str repr is robust to `from __future__ import annotations`.
        assert "PDBackendProtocol" in str(ann), (
            f"{fn_name} backend param annotated as {ann!r}, "
            "expected PDBackendProtocol"
        )
