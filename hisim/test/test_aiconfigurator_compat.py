import hisim.time_predictor.aiconfigurator as aiconfigurator


def test_install_aic_legacy_dtype_aliases(monkeypatch):
    enum_classes = (
        aiconfigurator.GEMMQuantMode,
        aiconfigurator.KVCacheQuantMode,
        aiconfigurator.FMHAQuantMode,
        aiconfigurator.MoEQuantMode,
    )

    for enum_cls in enum_classes:
        monkeypatch.delitem(enum_cls._member_map_, "bfloat16", raising=False)

    aiconfigurator._install_aic_legacy_dtype_aliases()
    aiconfigurator._install_aic_legacy_dtype_aliases()

    for enum_cls in enum_classes:
        assert enum_cls["bfloat16"] == enum_cls["float16"]


def test_backfill_aic_system_spec():
    system_spec = {"gpu": {"bfloat16_tc_flops": 123}}

    aiconfigurator._backfill_aic_system_spec(system_spec)

    assert system_spec["gpu"]["float16_tc_flops"] == 123
