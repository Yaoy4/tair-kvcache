"""Phase 5b — Picklable AIC predictor factory for BackendB workers.

BackendB's ``predictor_factory`` argument is invoked **inside** a freshly-
spawned worker process. The factory must therefore satisfy two constraints:

1. **Top-level / picklable.** No closures, no lambdas, no class-local methods.
2. **Cheap to pickle.** Heavy objects (DB handles, accelerator probe results)
   must be constructed in the child, not pickled across the boundary.

``AICPredictorFactory`` captures only the small, picklable inputs needed to
build an :class:`AIConfiguratorTimePredictor` and wrap it in an
:class:`AICPredictorAdapter`. The actual predictor + DB are constructed when
the worker calls ``factory()``.

Hardware lookup, predictor construction, and adapter wrapping all happen in
the child to avoid pickling any of those objects.
"""

from __future__ import annotations

import copy
import dataclasses
from typing import Callable, Optional

from hisim.spec import ModelInfo
from hisim.simulation.pd_aic_adapter import AICPredictorAdapter
from hisim.simulation.pd_config import RolePredictorConfig
from hisim.simulation.pd_factory import _build_role_sched_config
from hisim.simulation.types import SchedulerConfig


HWFactory = Callable[[str], object]
PredictorFactory = Callable[..., object]


def _default_hw_factory(name: str):
    # Lazy import keeps test envs that stub AcceleratorInfo unaffected.
    from hisim.spec import AcceleratorInfo

    hw = AcceleratorInfo.find_by_hw_name(name)
    if hw is None:
        raise ValueError(f"Unknown accelerator name {name!r}")
    return hw


def _default_predictor_factory(*args, **kwargs):
    from hisim.time_predictor import AIConfiguratorTimePredictor

    return AIConfiguratorTimePredictor(*args, **kwargs)


@dataclasses.dataclass
class AICPredictorFactory:
    """Picklable factory: ``factory()`` → adapter ready for BackendB.

    The instance holds only data classes / strings / floats so it survives
    the multiprocessing ``spawn`` boundary cleanly. Real heavyweight objects
    (HW probe, DB handle, predictor) are constructed in the child process
    when ``__call__`` runs.
    """

    model: ModelInfo
    base_sched_config: SchedulerConfig
    role: RolePredictorConfig
    # Optional overrides — useful for tests that want to inject stubs.
    hw_factory_dotted: Optional[str] = None
    predictor_factory_dotted: Optional[str] = None

    def __call__(self) -> AICPredictorAdapter:
        hw_factory = self._resolve(self.hw_factory_dotted, _default_hw_factory)
        predictor_factory = self._resolve(
            self.predictor_factory_dotted, _default_predictor_factory
        )
        hw = hw_factory(self.role.device_name)
        sched = _build_role_sched_config(
            copy.copy(self.base_sched_config), self.role, self.model
        )
        kwargs = {}
        if self.role.database_path is not None:
            kwargs["database_path"] = self.role.database_path
        kwargs["prefill_scale_factor"] = self.role.prefill_scale_factor
        kwargs["decode_scale_factor"] = self.role.decode_scale_factor
        base = predictor_factory(self.model, hw=hw, config=sched, **kwargs)
        return AICPredictorAdapter(base)

    @staticmethod
    def _resolve(dotted: Optional[str], default):
        if dotted is None:
            return default
        module_name, _, attr = dotted.rpartition(".")
        if not module_name:
            raise ValueError(
                f"factory override {dotted!r} must be a fully qualified name"
            )
        import importlib

        return getattr(importlib.import_module(module_name), attr)
