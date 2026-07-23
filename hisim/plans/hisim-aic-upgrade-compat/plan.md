# HiSim Compatibility with Newer aiconfigurator (AIC) Releases

**Branch:** `hisim-aic-upgrade-compat`
**Description:** Make HiSim's aiconfigurator (AIC) integration work against AIC's current
main-line architecture (Rust-backed `aiconfigurator-core`, renamed schema keys) instead of
being hard-pinned to the old `h20e-higher-acc` branch (AIC ~0.5.0).

## Background / Research Findings

**Where things currently stand (verified in this repo checkout):**

- HiSim (`/mnt/nfs02/users/tjiang/Gitrepo/tair-kvcache/hisim`, branch `Thjiang_Dev`) declares its
  AIC dependency in `pyproject.toml:13` as
  `aiconfigurator @ git+https://github.com/ai-dynamo/aiconfigurator.git@h20e-higher-acc`
  (commit `9f744a19`), installed in the HiSim venv as `aiconfigurator-0.5.0`
  (`hisim/lib/python3.13/site-packages/aiconfigurator-0.5.0.dist-info`).
- The AIC repo (`/mnt/nfs02/users/tjiang/Gitrepo/aiconfigurator`) is on `main` at commit
  `e0735ccc` (still reports `version = "0.5.0"` in `pyproject.toml`, but is functionally much
  further along — see below). A `release/0.9.0` branch and a later "bump to 0.9.0" commit
  (`3889ee52`) also exist locally.
- Per the team's own debugging notes (`tair-kvcache/目前遇到的问题.md` and
  `tair-kvcache/260721-实验测试统一参数与指令-蒋天颢.md`, both already in this repo, dated
  2026-07-17/2026-07-21), a colleague already attempted this exact upgrade:
  - The AIC "main" they pulled reports version **0.10**; the perf-database tables moved from
    **`.txt` to Parquet**, and a separate compiled package **`aiconfigurator-core`** (Rust,
    built via `maturin`) is now required — it is not on PyPI (`pip install aiconfigurator-core==0.10.0`
    fails) and must be built from AIC's `rust/aiconfigurator-core` source.
  - After building `aiconfigurator-core` locally, the run failed with:
    `AttributeError: 'PerfDatabase' object has no attribute '_nearest_1d_point_helper'`.
  - **Root cause located in this session:** HiSim itself monkey-patches this *private* AIC
    method in `hisim/src/hisim/time_predictor/aiconfigurator.py:443-451`:
    ```python
    db_nearest_1d_point_helper = database._nearest_1d_point_helper
    ...
    database._nearest_1d_point_helper = wrapped_nearest_1d_point_helper
    ```
    This still exists in AIC's current local `main` (e0735ccc, 20 references in
    `perf_database.py`), so today's pinned setup works — but it is exactly the kind of
    private-API dependency that breaks the moment AIC's Rust-core migration removes/relocates
    the Python implementation, which is what happened in the colleague's "0.10" build.
  - The team's practical workaround (2026-07-21 notes) was to pin to AIC commit `e0735ccc`
    directly (referred to as `aiconfigurator-e0735cc` in their `PYTHONPATH`/`database_path`),
    which already contains other forward-compatible changes (see next point) but predates the
    Rust-core/Parquet migration — i.e. a middle ground, not a real fix.
- Separately (found independently in this session via git diff `v0.5.0` vs `release/0.9.0`/
  current `main`), the following AIC surface changes already occurred **before** the Rust-core
  cutover and HiSim only partially/inconsistently accounts for them:
  - `gpu.float16_tc_flops` → `gpu.bfloat16_tc_flops` in system YAML files and in
    `perf_database.py`'s SOL-math (plain dict subscript, **no fallback key**). Confirmed via
    `git diff v0.5.0 origin/release/0.9.0 -- src/aiconfigurator/systems/b200_sxm.yaml` and
    `perf_database.py`. At current `main` HEAD (`e0735ccc`) the key is **already**
    `bfloat16_tc_flops`. The historical results folder
    `/mnt/nfs02/users/tjiang/Gitrepo/sglang/HiSim_dp1_vs_dp2_20260622/custom_systems/rtx_pro_6000_server.yaml`
    still uses the **old** `float16_tc_flops` key (its own header comment even says
    "Adapted for installed aiconfigurator venv (float16_tc_flops key naming)") — this file is a
    historical artifact and should not be edited, but any *currently active* custom system YAML
    used for new runs must use the new key or it will `KeyError` against modern AIC.
  - AIC's `main` now ships an **official** `src/aiconfigurator/systems/rtx_pro_6000_server.yaml`
    (added by upstream PR "fix: add RTX PRO support matrix rows (#1071)"), which did not exist
    when HiSim's custom copy was first written. The team's 2026-07-21 config already points
    `database_path` at AIC's own `systems/` directory rather than HiSim's custom copy.
  - `aiconfigurator.sdk.models` changed from a single file to a package (`models/`), but
    `BaseModel`/`get_model` are still re-exported — **not** a break for HiSim's current usage.
  - `SupportedModels` removed from `common.py`, `get_system_config_path` renamed to
    `get_systems_paths` — HiSim already has try/except fallbacks for both
    (`aiconfigurator.py:21-37`, `112-120`) — **not** a break.
  - `requires-python` bumped from `>=3.9` to `>=3.10` — HiSim's venv is Python 3.13, unaffected.

**Net assessment:** the user's statement is correct — HiSim's `main`/`Thjiang_Dev` branch does
not truly support current AIC (colloquially "0.9.0"/"0.10"). The single concrete, reproduced
blocker is the `_nearest_1d_point_helper` private-API monkeypatch once AIC's Rust core is used;
the `*_tc_flops` key rename is a second, independent, already-diffed break for anyone still using
the old custom YAML naming.

**[NEEDS CLARIFICATION]** The exact target AIC version to pin going forward: locally we only
have `main`@`e0735ccc` and `release/0.9.0`@`3889ee52` (both still using the pre-Rust-core, `.txt`
Python `PerfDatabase`). The colleague's "0.10" with the mandatory `aiconfigurator-core` Rust
package and Parquet tables was pulled from upstream and is not present in this local clone —
confirming its exact tag/commit requires a `git fetch` against
`https://github.com/ai-dynamo/aiconfigurator` (not done in this planning session, to avoid
mutating repo state without approval). Step 1 below assumes we will fetch and target that
upstream "0.10" commit; if the team instead wants to standardize on `release/0.9.0` (no Rust
core required yet), Steps 2–3 still apply but Step 1's Rust build sub-tasks can be skipped.

## Goal

Remove HiSim's dependency on AIC's private/removed Python internals and stale system-YAML key
names so that HiSim can run against current-generation AIC (Rust-core + Parquet) without manual
per-machine workarounds, instead of being permanently pinned to an old custom branch/commit.

## Implementation Steps

### Step 1: Reproduce the upgrade and pin down the exact target AIC version
**Files:** none (environment/investigation only); new `tair-kvcache/hisim/docs/develop/aic_upgrade_notes.md` to record findings.
**What:** `git fetch` the AIC remote to find the actual "0.10" tag/commit the colleague built
(or confirm `release/0.9.0` is the intended target). Reproduce their steps in a clean venv:
`pip install 'maturin>=1.12,<2'` then `pip install -e <aic_repo>/rust/aiconfigurator-core`,
set `PYTHONPATH` to include both HiSim `src` and AIC `src`, and capture the exact failure
(`AttributeError: ... _nearest_1d_point_helper`) as a baseline before code changes. Record the
resolved version numbers (`pip show aiconfigurator aiconfigurator-core`) in the new notes file.
**Testing:** `python -c "import aiconfigurator_core, aiconfigurator"` succeeds; document the
AttributeError baseline (expected to still fail — this step only pins down versions/repro, no
fix yet).

### Step 2: Make the `_nearest_1d_point_helper` monkeypatch version-safe
**Files:** `hisim/src/hisim/time_predictor/aiconfigurator.py` (~lines 440-451);
`hisim/tests/` (new/extended unit test).
**What:** Guard the read/patch with `hasattr(database, "_nearest_1d_point_helper")`, matching
the try/except style already used elsewhere in this file (`SupportedModels`,
`get_system_config_path`/`get_systems_paths`, `_enum_member`). When absent, skip the
wrap-and-cache step entirely (log a debug note) rather than raising. If the wrapper exists for
performance (memoization), check whether the Rust-backed lookup path needs an equivalent cache;
if so add one guarded by the same `hasattr` check instead of assuming the private method exists.
**Testing:** New unit test constructs a stub `PerfDatabase`-like object without
`_nearest_1d_point_helper` and asserts `get_perf_model`/session construction completes without
`AttributeError`. Re-run existing `pytest tests/` (and `test/`) to confirm the old code path
(attribute present) is unchanged.

### Step 3: Fix system YAML key naming and adopt upstream `rtx_pro_6000_server.yaml` where applicable
**Files:** any HiSim-maintained custom system YAML currently in active use (locate via
`grep -rn "float16_tc_flops" tair-kvcache/hisim`), plus any `tools/*.json` configs whose
`database_path` points at a HiSim-local copy instead of AIC's own `systems/` dir. Do **not**
modify the historical `sglang/HiSim_dp1_vs_dp2_20260622/` results artifacts.
**What:** Rename `float16_tc_flops` → `bfloat16_tc_flops` in any actively-used custom YAML still
on the old name. Compare field-by-field against AIC's now-official
`src/aiconfigurator/systems/rtx_pro_6000_server.yaml` and decide whether to switch
`database_path` to point at AIC's own systems directory (as the 2026-07-21 config already does)
or keep a HiSim-maintained override — document the choice and why (e.g., HiSim may intentionally
override empirical scaling factors).
**Testing:** Run a minimal sweep (reuse `run_minimal_sweep.sh` pattern from
`HiSim_dp1_vs_dp2_20260622/`) against the updated config; confirm no `KeyError` on the
`*_tc_flops` keys and that TTFT/TPOT outputs are in the same ballpark as historical results.

### Step 4: Update the pinned dependency and build/install instructions
**Files:** `hisim/pyproject.toml`, `hisim/README.md` / `README_zh.md`, new changelog entry
(e.g. `hisim/docs/develop/`).
**What:** Replace the `@h20e-higher-acc` git pin with the version confirmed in Step 1
(exact commit/tag), add the `aiconfigurator-core` build step (maturin-based, since not on PyPI)
to setup docs, and document the `_nearest_1d_point_helper` fix and the `*_tc_flops` rename as a
short "upgrading AIC" note for future contributors.
**Testing:** Fresh-venv bootstrap following the updated instructions succeeds with no manual
workarounds; run full `pytest` suite; run one end-to-end HiSim sweep as smoke validation.

### Step 5 (optional / separate follow-up — confirm before doing): Regenerate dp1 vs dp2 comparison results
**Files:** new dated results folder under `sglang/` (e.g. `HiSim_dp1_vs_dp2_<new-date>/`) —
leave the original `HiSim_dp1_vs_dp2_20260622/` untouched.
**What:** Re-run `compare_dp1_dp2.sh` / `run_hisim_dp1_sweep.sh` against the upgraded AIC stack,
producing fresh figures/CSV, and note any numerical deltas vs. the 2026-06-22 baseline.
**Testing:** Sanity-check MAPE/error-breakdown figures aren't wildly different from the
historical baseline; flag large deviations for review instead of silently accepting them. This
step is a larger, potentially long-running effort and should be scheduled independently of
Steps 1–4.
