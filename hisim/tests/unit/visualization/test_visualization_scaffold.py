"""V0 scaffold smoke test.

Verifies the visualization package and its placeholder submodules import
cleanly. No behaviour is asserted beyond import success.
"""

from __future__ import annotations


def test_visualization_package_imports() -> None:
    import hisim.visualization as viz

    assert viz.__doc__ is not None


def test_visualization_submodules_import() -> None:
    from hisim.visualization import sweep_data, sweep_figures, sweep_report

    for module in (sweep_data, sweep_figures, sweep_report):
        assert module.__doc__ is not None
