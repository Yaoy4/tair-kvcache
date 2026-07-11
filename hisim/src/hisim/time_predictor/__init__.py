from hisim.time_predictor.base import (
    InferTimePredictor,
    FakeRequest,
    ScheduleBatch,
)


__all__ = (
    "FakeRequest",
    "ScheduleBatch",
    "InferTimePredictor",
    "AIConfiguratorTimePredictor",
)


def __getattr__(name):
    if name == "AIConfiguratorTimePredictor":
        from hisim.time_predictor.aiconfigurator import (
            AIConfiguratorTimePredictor,
        )

        return AIConfiguratorTimePredictor
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
