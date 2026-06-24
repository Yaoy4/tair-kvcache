import ast
import inspect

from hisim.simulation.pd_types import PDRequestState, RequestPhase


def test_request_phase_order_and_legal_path():
    phases = list(RequestPhase)
    assert phases == [
        RequestPhase.WAITING_PREFILL,
        RequestPhase.RUNNING_PREFILL,
        RequestPhase.KV_TRANSIT,
        RequestPhase.WAITING_DECODE,
        RequestPhase.RUNNING_DECODE,
        RequestPhase.FINISHED,
    ]

    legal_path = {
        RequestPhase.WAITING_PREFILL: RequestPhase.RUNNING_PREFILL,
        RequestPhase.RUNNING_PREFILL: RequestPhase.KV_TRANSIT,
        RequestPhase.KV_TRANSIT: RequestPhase.WAITING_DECODE,
        RequestPhase.WAITING_DECODE: RequestPhase.RUNNING_DECODE,
        RequestPhase.RUNNING_DECODE: RequestPhase.FINISHED,
    }

    for current_phase, next_phase in legal_path.items():
        assert phases[phases.index(current_phase) + 1] == next_phase


def test_pd_request_state_initialization_is_deterministic():
    lhs = PDRequestState(rid="r-1", arrival_time=1.25)
    rhs = PDRequestState(rid="r-1", arrival_time=1.25)

    assert lhs == rhs
    assert lhs.phase == RequestPhase.WAITING_PREFILL
    assert lhs.prefill_start_time is None
    assert lhs.prefill_end_time is None
    assert lhs.kv_ready_time is None
    assert lhs.decode_start_time is None
    assert lhs.decode_step_count == 0
    assert lhs.current_past_kv_length == 0
    assert lhs.output_length == 0


def test_pd_types_has_no_sglang_dependency():
    module_src = inspect.getsource(inspect.getmodule(PDRequestState))
    module_ast = ast.parse(module_src)

    imported_modules = set()
    for node in ast.walk(module_ast):
        if isinstance(node, ast.Import):
            for alias in node.names:
                imported_modules.add(alias.name)
        if isinstance(node, ast.ImportFrom) and node.module:
            imported_modules.add(node.module)

    assert not any(name.startswith("sglang") for name in imported_modules)
