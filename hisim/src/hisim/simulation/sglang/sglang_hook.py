import torch
import time
from dataclasses import asdict
from collections import defaultdict
import json
import os
import heapq
import importlib
import threading
from queue import Empty

from hisim.utils import get_logger

from hisim.hook import BaseHook
from hisim.simulation.types import (
    MockSimulationMode,
    RequestStats,
)
from hisim.hook.utils import get_obj_from_args
from hisim.utils.json import CustomJsonEncoder
from hisim.simulation.manager import StateManager, ConfigManager, Envs
from hisim.time_predictor import (
    InferTimePredictor,
    FakeRequest,
    ScheduleBatch as HisimScheduleBatch,
)
from hisim.simulation.sglang.sglang_mock_class import (
    MockReqToTokenPool,
    MockTokenToKVPool,
    MockTokenToKVPoolAllocator,
    MockPagedTokenToKVPoolAllocator,
    MockTokenToKVPoolHost,
    MockHiCacheStorage,
)
from hisim.simulation.utils import (
    calc_metrics,
    estimate_kv_cache_pool_capacity,
)
from hisim.simulation.sglang.version import VersionDispatcher


logger = get_logger("hisim")


class C_EngineHook(BaseHook):
    HOOK_CLASS_NAME = "Engine"
    HOOK_MODULE_NAME = "sglang.srt.entrypoints.engine"

    @classmethod
    def hook(cls, target):
        def hook_clear_hicache_storage(self):
            return self.loop.run_until_complete(
                self.tokenizer_manager.clear_hicache_storage()
            )

        target.clear_hicache_storage = hook_clear_hicache_storage


class C_TokenizerManagerHook(BaseHook):
    HOOK_CLASS_NAME = "TokenizerManager"
    HOOK_MODULE_NAME = "sglang.srt.managers.tokenizer_manager"

    @classmethod
    def hook(cls, target):
        original_send_one_request = target._send_one_request

        # When running with blocking mode, send the created time to schedule.
        def wrapped_send_one_request(self, obj, tokenized_obj, created_time):
            if obj.__class__.__name__ == "GenerateReqInput":
                if (
                    tokenized_obj.sampling_params.custom_params is not None
                    and "simulation" in tokenized_obj.sampling_params.custom_params
                ):
                    tokenized_obj.sampling_params.custom_params["simulation"][
                        "server_created_time"
                    ] = created_time
            return original_send_one_request(self, obj, tokenized_obj, created_time)

        target._send_one_request = wrapped_send_one_request


class C_ModelRunnerHook(BaseHook):
    HOOK_CLASS_NAME = "ModelRunner"
    HOOK_MODULE_NAME = "sglang.srt.model_executor.model_runner"

    @classmethod
    def hook(cls, target):
        _version_dispatcher = VersionDispatcher()

        def override_initialize(self, *args, **kwargs):
            class MockModel:
                def forward(self):
                    pass

            self.model = MockModel()

            self.dtype = self.model_config.dtype
            self.kv_cache_dtype = (
                self.dtype
            )  # FIXME: get kv cache dtype from server args

            model = ConfigManager.get_model_info(self.model_config.hf_config.__dict__)
            hw = ConfigManager.get_accelerator_info()
            config = ConfigManager.get_scheduler_config(
                self.server_args.__dict__,
                "sglang",
                self.model_config.hf_config.__dict__,
            )

            assert model is not None and hw is not None and config is not None

            if self.server_args.max_total_tokens is not None:
                self.max_total_num_tokens = self.server_args.max_total_tokens
            else:
                self.max_total_num_tokens = estimate_kv_cache_pool_capacity(
                    model, hw, config
                )

            if hasattr(self, "page_size") and self.page_size > 1:
                self.max_total_num_tokens = (
                    self.max_total_num_tokens // self.page_size * self.page_size
                )

            max_num_reqs = min(
                max(
                    int(self.max_total_num_tokens / model.max_seq_len * 512),
                    2048,
                ),
                4096,
            )
            logger.info(
                f"Model runner initialized with {self.max_total_num_tokens} tokens. Maximum number of requests: {max_num_reqs}"
            )

            model_has_mtp_layers = (
                self.model_config.num_nextn_predict_layers is not None
            )
            model_num_layers = (
                self.model_config.num_nextn_predict_layers
                if self.is_draft_worker and model_has_mtp_layers
                else max(
                    self.model_config.num_hidden_layers,
                    self.model_config.num_attention_layers,
                )
            )
            self.start_layer = getattr(self.model, "start_layer", 0)
            self.end_layer = getattr(self.model, "end_layer", model_num_layers)
            self.num_effective_layers = self.end_layer - self.start_layer

            self.req_to_token_pool = MockReqToTokenPool(
                size=max_num_reqs,
                max_context_len=model.max_seq_len,
                device=self.device,
                enable_memory_saver=False,
            )

            self.token_to_kv_pool = MockTokenToKVPool(
                self.max_total_num_tokens,
                page_size=self.page_size,
                dtype=self.kv_cache_dtype,
                head_num=self.model_config.get_num_kv_heads(
                    1  # get_attention_tp_size()
                ),
                head_dim=self.model_config.head_dim,
                layer_num=self.num_effective_layers,
                device=self.device,
                enable_memory_saver=self.server_args.enable_memory_saver,
                start_layer=self.start_layer,
                end_layer=self.end_layer,
            )

            if self.page_size == 1:
                self.token_to_kv_pool_allocator = MockTokenToKVPoolAllocator(
                    size=self.max_total_num_tokens,
                    page_size=1,
                    dtype=self.kv_cache_dtype,
                    device=self.device,
                    kvcache=self.token_to_kv_pool,
                    need_sort=False,
                )
            else:
                self.token_to_kv_pool_allocator = MockPagedTokenToKVPoolAllocator(
                    size=self.max_total_num_tokens,
                    page_size=self.page_size,
                    dtype=self.kv_cache_dtype,
                    device=self.device,
                    kvcache=self.token_to_kv_pool,
                    need_sort=False,
                )

            # self.init_memory_pool(50)
            self.attn_backend = None
            self.graph_mem_usage = 0
            self.weight_load_mem_usage = 10

            self.max_running_requests = min(
                (
                    self.max_total_num_tokens // 2
                    if self.server_args.max_running_requests is None
                    else self.server_args.max_running_requests
                    // (
                        self.server_args.dp_size
                        if self.server_args.enable_dp_attention
                        else 1
                    )
                ),
                self.req_to_token_pool.size,
            )

            return

        def wrapped_forward_v1(self, *args, **kwargs):
            batch = args[0]
            from sglang.srt.layers.logits_processor import LogitsProcessorOutput

            output = LogitsProcessorOutput(
                next_token_logits=torch.empty(
                    size=(batch.batch_size, self.model_config.vocab_size),
                    device=self.device,
                )
            )

            return output, False

        _version_dispatcher.register_method(
            "forward", ["0.5.6", "0.5.6.post1", "0.5.6.post2"], wrapped_forward_v1
        )

        def wrapped_forward_v2(self, *args, **kwargs):
            from sglang.srt.model_executor.model_runner import ModelRunnerOutput

            output, _ = wrapped_forward_v1(self, *args, **kwargs)
            return ModelRunnerOutput(
                logits_output=output,
                can_run_graph=False,
                expert_distribution_metrics=None,
            )

        _version_dispatcher.register_method(
            "forward", ["0.5.7", "0.5.8", "0.5.8.post1", "0.5.9"], wrapped_forward_v2
        )

        def wrapped_sample(self, *args, **kwargs):
            logits = args[0]
            ids = torch.ones(
                size=(logits.next_token_logits.shape[0],),
                device=self.device,
                dtype=torch.int64,
            )
            return ids

        def wrapped_compute_logprobs_only(*args, **kwargs):
            return None

        target.initialize = override_initialize
        target.forward = _version_dispatcher.get_compat_method("forward")
        target.sample = wrapped_sample
        target.compute_logprobs_only = wrapped_compute_logprobs_only


class C_HiCacheController(BaseHook):
    HOOK_CLASS_NAME = "HiCacheController"
    HOOK_MODULE_NAME = "sglang.srt.managers.cache_controller"

    KV_CACHE_BYTES: int = None
    DISK_READ_BANDWIDTH_BYTES: float = None
    DISK_WRITE_BANDWIDTH_BYTES: float = None

    @staticmethod
    def calc_prefetch_pages(
        required_pages: int, page_size_byte: int, max_dur: float, bandwidth: float
    ) -> tuple[float, float]:
        _prefetch_dur = required_pages * page_size_byte / bandwidth
        if _prefetch_dur > max_dur:
            _completed_pages = max(max_dur * bandwidth / page_size_byte, 1)
            return _completed_pages, max_dur
        else:
            return required_pages, _prefetch_dur

    @classmethod
    def hook(cls, target):
        def override_backup_thread_func(self, *args, **kwargs):
            # Async thread: perform no action
            # The action will be performed by `handle_backup_operation`
            pass

        def override_prefetch_thread_func(self, *args, **kwargs):
            # Async thread: perform no action
            # The action will be performed by `handle_prefetch_operation`
            pass

        def handle_backup_operation(self):
            if not self.enable_storage:
                return
            while True:
                try:
                    operation = self.backup_queue.get(block=False)
                    if operation is None:
                        return

                    if not self.backup_skip:
                        self._page_backup(operation)
                    # TODO: Track the backup operation according to the global clock
                    self.ack_backup_queue.put(operation)

                except Empty:
                    return

        def handle_prefetch_operation(self):
            if not self.enable_storage:
                return

            if C_HiCacheController.KV_CACHE_BYTES is None:
                C_HiCacheController.KV_CACHE_BYTES = ConfigManager.get_kv_cache_bytes()
            if C_HiCacheController.DISK_READ_BANDWIDTH_BYTES is None:
                C_HiCacheController.DISK_READ_BANDWIDTH_BYTES = (
                    ConfigManager.get_platform_config().disk_read_bandwidth
                )

            # TODO: Overlap schedule
            remain_dur = StateManager.get_current_inference_dur()

            chunked_prefetch_operation = getattr(
                self, "chunked_prefetch_operation", None
            )
            if chunked_prefetch_operation is not None:
                operation = chunked_prefetch_operation["operation"]
                storage_hit_count = chunked_prefetch_operation["storage_hit_count"]
                completed_tokens, prefetch_dur = (
                    C_HiCacheController.calc_prefetch_pages(
                        (storage_hit_count - operation.completed_tokens),
                        C_HiCacheController.KV_CACHE_BYTES,
                        remain_dur,
                        C_HiCacheController.DISK_READ_BANDWIDTH_BYTES,
                    )
                )
                if completed_tokens < storage_hit_count - operation.completed_tokens:
                    operation.completed_tokens += completed_tokens
                    remain_dur = 0
                else:
                    operation.completed_tokens = int(storage_hit_count)
                    operation.mark_terminate()
                    remain_dur -= prefetch_dur
                    setattr(self, "chunked_prefetch_operation", None)
                    # Release host memory after current operation is finished
                    self.append_host_mem_release(
                        operation.host_indices[storage_hit_count:]
                    )
                # update request states
                req_stats = C_SchedulerHook.REQUEST_STATS[operation.request_id]
                req_stats.prefetch_complete_tokens = operation.completed_tokens

            while remain_dur > 0:
                try:
                    operation = self.prefetch_queue.get(block=False)
                    if operation is None:
                        return

                    hash_value, storage_hit_count = self._storage_hit_query(operation)
                    # not to prefetch if not enough benefits
                    if (
                        self.prefetch_threshold is not None
                        and storage_hit_count < self.prefetch_threshold
                    ):
                        operation.mark_terminate()
                        self.append_host_mem_release(operation.host_indices)
                        continue

                    operation.hash_value = hash_value[
                        : (storage_hit_count // self.page_size)
                    ]
                    storage_hit_count = (
                        storage_hit_count // self.page_size * self.page_size
                    )

                    completed_tokens, prefetch_dur = (
                        C_HiCacheController.calc_prefetch_pages(
                            storage_hit_count,
                            C_HiCacheController.KV_CACHE_BYTES,
                            remain_dur,
                            C_HiCacheController.DISK_READ_BANDWIDTH_BYTES,
                        )
                    )
                    if completed_tokens < storage_hit_count:
                        # Continue to prefetch data next time.
                        operation.completed_tokens = int(completed_tokens)
                        setattr(
                            self,
                            "chunked_prefetch_operation",
                            {
                                "operation": operation,
                                "storage_hit_count": storage_hit_count,
                            },
                        )
                        remain_dur = 0
                    else:
                        operation.completed_tokens = int(
                            storage_hit_count // self.page_size * self.page_size
                        )
                        # TODO: Track the prefetch operation according to the global clock
                        operation.mark_terminate()
                        remain_dur -= prefetch_dur
                    # update request states
                    req_stats = C_SchedulerHook.REQUEST_STATS[operation.request_id]
                    req_stats.prefetch_complete_tokens = operation.completed_tokens
                    # Release host memory after current operation is finished
                    self.append_host_mem_release(
                        operation.host_indices[storage_hit_count:]
                    )

                except Empty:
                    return

        def override_generic_page_set(
            self, hash_values, host_indices, extra_info=None
        ) -> bool:
            # Always pass extra_info to storage_backend.
            data = [
                self.mem_pool_host.get_data_page(host_indices[i * self.page_size])
                for i in range(len(hash_values))
            ]
            return self.storage_backend.batch_set(hash_values, data, extra_info)

        target.prefetch_thread_func = override_prefetch_thread_func
        target.backup_thread_func = override_backup_thread_func
        target.handle_backup_operation = handle_backup_operation
        target.handle_prefetch_operation = handle_prefetch_operation
        target._generic_page_set = override_generic_page_set


class C_HiRadixCacheHook(BaseHook):
    HOOK_CLASS_NAME = "HiRadixCache"
    HOOK_MODULE_NAME = "sglang.srt.mem_cache.hiradix_cache"

    @classmethod
    def hook(cls, target):
        original_check_hicache_events = target.check_hicache_events
        original_reset = target.reset

        def wrapped_reset(self):
            if hasattr(self, "cache_controller"):
                self.cache_controller.handle_backup_operation()
            original_reset(self)

        def override_init(self, params, server_args):
            if server_args.hicache_io_backend == "direct":
                # FIXME: move this logic into server_args parsing
                if server_args.hicache_mem_layout == "page_first":
                    server_args.hicache_mem_layout = "page_first_direct"
                    logger.warning(
                        "Page first layout is not supported with direct IO backend, switching to page first direct layout"
                    )

            self.page_size = params.page_size
            self.kv_cache = params.token_to_kv_pool_allocator.get_kvcache()
            # Replace the host pool
            self.token_to_kv_pool_host = MockTokenToKVPoolHost(
                self.kv_cache,
                server_args.hicache_ratio,
                server_args.hicache_size,
                self.page_size,
                server_args.hicache_mem_layout,
                pin_memory=False,
                device="cpu",
            )

            self.tp_group = params.tp_cache_group
            self.tp_world_size = torch.distributed.get_world_size(group=self.tp_group)
            self.enable_storage = server_args.hicache_storage_backend is not None
            self.enable_storage_metrics = self.enable_storage and params.enable_metrics

            (
                extra_config,
                prefetch_threshold,
                prefetch_timeout_base,
                prefetch_timeout_per_ki_token,
                hicache_storage_pass_prefix_keys,
            ) = self._parse_storage_backend_extra_config(
                server_args.hicache_storage_backend_extra_config
            )
            self.prefetch_threshold = prefetch_threshold
            self.prefetch_timeout_base = prefetch_timeout_base
            self.prefetch_timeout_per_page = (
                self.page_size / 1024 * prefetch_timeout_per_ki_token
            )
            self.hicache_storage_pass_prefix_keys = hicache_storage_pass_prefix_keys
            # TODO: support more timeout check functions
            self.is_prefetch_timeout = self._prefetch_timeout_check_linear_func
            self.prefetch_stop_policy = server_args.hicache_storage_prefetch_policy

            HiCacheController = getattr(
                importlib.import_module("sglang.srt.managers.cache_controller"),
                "HiCacheController",
            )
            StorageMetricsCollector = getattr(
                importlib.import_module("sglang.srt.metrics.collector"),
                "StorageMetricsCollector",
            )

            self.load_cache_event = threading.Event()
            self.cache_controller = HiCacheController(
                params.token_to_kv_pool_allocator,
                self.token_to_kv_pool_host,
                self.page_size,
                self.tp_group,
                load_cache_event=self.load_cache_event,
                write_policy=server_args.hicache_write_policy,
                io_backend=server_args.hicache_io_backend,
                storage_backend=server_args.hicache_storage_backend,
                prefetch_threshold=self.prefetch_threshold,
                model_name=server_args.served_model_name,
                storage_backend_extra_config=extra_config,
            )
            if self.enable_storage_metrics:
                # TODO: support pp
                labels = {
                    "storage_backend": server_args.hicache_storage_backend,
                    "tp_rank": self.cache_controller.tp_rank,
                    "dp_rank": self.cache_controller.dp_rank,
                }
                self.storage_metrics_collector = StorageMetricsCollector(labels=labels)

            # Record the nodes with ongoing write-through
            self.ongoing_write_through = {}
            # Record the node segments with ongoing load-back
            self.ongoing_load_back = {}
            # Record the ongoing prefetch requests
            self.ongoing_prefetch = {}
            self.ongoing_backup = {}
            # TODO: Dynamically adjust the threshold
            self.write_through_threshold = (
                1 if server_args.hicache_write_policy == "write_through" else 2
            )
            self.load_back_threshold = 10
            # Version: 0.5.9
            self.prefetch_loaded_tokens_by_reqid: dict[str, int] = {}
            self.evictable_host_leaves = set()
            # super().__init__(params=params)
            target.__mro__[1].__init__(self, params=params)

        def wrapped_check_hicache_events(self, *args, **kwargs):
            # Call operation handler first.
            self.cache_controller.handle_backup_operation()
            self.cache_controller.handle_prefetch_operation()
            return original_check_hicache_events(self, *args, **kwargs)

        target.__init__ = override_init
        target.check_hicache_events = wrapped_check_hicache_events
        target.reset = wrapped_reset


class C_StorageBackendFactory(BaseHook):
    HOOK_CLASS_NAME = "StorageBackendFactory"
    HOOK_MODULE_NAME = "sglang.srt.mem_cache.storage.backend_factory"

    @classmethod
    def hook(cls, target):
        def override_create_backend(cls, *args, **kwargs):
            logger.info("Creating hijacked cache storage backend.")
            return MockHiCacheStorage()

        target.create_backend = override_create_backend


class C_SchedulerHook(BaseHook):
    HOOK_CLASS_NAME = "Scheduler"
    HOOK_MODULE_NAME = "sglang.srt.managers.scheduler"

    INFERENCE_PREDICTOR: InferTimePredictor = None

    REQUEST_STATS: dict[str, RequestStats] = defaultdict(RequestStats)
    ITERATION_STATS: list[dict] = []
    LAST_CPU_TS: float = 0
    LAST_FLUSH_TS: float = 0
    HISIM_BATCH: HisimScheduleBatch = None

    OVERLAP_SCHEDULE: bool = False

    # PD disaggregation (Phase 2b): populated in wrapped_init when the
    # config's `disagg.enabled` is True. Stays None for the aggregated path.
    PD_BACKEND = None  # Optional[BackendA]
    # rid -> PDRequestState; populated lazily as sglang surfaces requests.
    PD_REQUEST_STATES: dict = {}
    # rid -> token completion time for the most recent PD decode scheduling
    # pass. Under per_replica_queue, different decode replicas can finish at
    # different times within one SGLang scheduler iteration.
    PD_BATCH_TOKEN_TIMES: dict[str, float] = {}
    # P1 dual-clock decoupling: the single global clock serialised prefill and
    # decode (a prefill batch + its KV transfer froze every in-flight decode in
    # virtual time). The decode role clock advances independently of prefill —
    # only by decode batches — and reconciles at the KV handoff via
    # sync_decode_start. It self-anchors to real kv_ready times, so 0 init is
    # valid in both OFFLINE and BLOCKING modes.
    #
    # Neither role's "engine free" floor is a hand-tracked scalar (it used to
    # be PD_PREFILL_CLOCK / PD_DECODE_CLOCK, both removed): a scalar can only
    # remember one specific replica's state, which forces later batches/steps
    # to wait for THAT replica even when a different one has been idle the
    # whole time. Instead each floor is read directly off the backend's own
    # pool state via PD_BACKEND.earliest_pool_time("prefill" | "decode"),
    # which already knows -- per backend, per decode_queue_mode -- whether
    # replicas are independent (use the pool minimum) or share one combined
    # cohort each step (use the pool maximum, i.e. the last step's actual
    # completion, to avoid starting a request's next step before its own
    # previous step finished).
    #
    # Max end time of the most recent PD decode scheduling pass, on the decode
    # role clock. process_batch_result records decode tokens from
    # PD_BATCH_TOKEN_TIMES where available and falls back to this value.
    PD_LAST_DECODE_STEP_END: float = 0.0
    # Per-request service span accumulated on the prefill role timeline:
    # (kv_ready_time - prefill_start_time). Used by closed-loop first-token
    # TTFT accounting to drop synthetic t=0 cap queueing.
    PD_PREFILL_KV_SERVICE: dict[str, float] = {}
    # Accumulated chunk token counts for chunked-prefill requests. Tracks the
    # running sum of extend_input_len across extend batches so full prompt
    # length is known at finalize time when origin_input_ids is unavailable.
    PD_CHUNK_ACCUM: dict = {}
    # Closed-loop TTFT accounting switch (enabled via disagg.closed_loop_ttft in
    # HISIM config, with HISIM_PD_CLOSED_LOOP env as an override fallback).
    PD_CLOSED_LOOP: bool = False

    SIM_MODE = MockSimulationMode(Envs.simulation_mode())
    OFFLINE_RECV_ALL_REQUEST: bool = False
    FUTURE_QUEUE: list[
        tuple[float, int, RequestStats]
    ] = []  # tuple(created time, salt, request)

    SCHEDULE_REQ_STATS = []

    @classmethod
    def hook(cls, target):
        original_init = target.__init__
        original_recv_requests = target.recv_requests
        original_get_new_batch_prefill = target.get_new_batch_prefill
        original_update_running_batch = target.update_running_batch
        original_run_batch = target.run_batch
        original_process_batch_result = target.process_batch_result
        original_event_loop_normal = target.event_loop_normal

        def override_event_loop_overlap(self, *args, **kwargs):
            # To reduce the complexity of the simulation, the overlapping schedule is not needed.
            return original_event_loop_normal(self, *args, **kwargs)

        def wrapped_init(self, *args, **kwargs):
            # Disable overlap schedule
            server_args = get_obj_from_args(
                "sglang.srt.server_args.ServerArgs", *args, **kwargs
            )
            C_SchedulerHook.OVERLAP_SCHEDULE = not getattr(
                server_args, "disable_overlap_schedule", False
            )
            setattr(server_args, "disable_overlap_schedule", True)
            logger.debug(
                f"Overlap schedule simulation mode: {C_SchedulerHook.OVERLAP_SCHEDULE}."
            )

            disagg_cfg = ConfigManager.get_disagg_config()
            if disagg_cfg.enabled and disagg_cfg.decode is not None:
                pd_capacity = min(
                    disagg_cfg.prefill_admission_capacity(),
                    disagg_cfg.decode_admission_capacity(),
                )
                configured_capacity = getattr(
                    server_args, "max_running_requests", None
                )
                if configured_capacity is not None:
                    pd_capacity = min(
                        int(configured_capacity), int(pd_capacity)
                    )
                setattr(server_args, "max_running_requests", pd_capacity)

            original_init(self, *args, **kwargs)
            cfg_closed_loop = False
            try:
                with open(Envs.config_path(), "r", encoding="utf-8") as f:
                    cfg_closed_loop = bool(
                        json.load(f)
                        .get("disagg", {})
                        .get("closed_loop_ttft", False)
                    )
            except Exception:
                cfg_closed_loop = False
            C_SchedulerHook.PD_CLOSED_LOOP = (
                cfg_closed_loop or Envs.pd_closed_loop()
            )
            logger.info(
                "PD closed-loop TTFT mode: %s",
                C_SchedulerHook.PD_CLOSED_LOOP,
            )
            model = ConfigManager.get_model_info(
                self.model_config.hf_config.__dict__
            )
            hw = ConfigManager.get_accelerator_info()
            sched_config = ConfigManager.get_scheduler_config(
                self.server_args.__dict__,
                "sglang",
                self.model_config.hf_config.__dict__,
            )
            ConfigManager.set_scheduler_config(sched_config)
            ConfigManager.set_model_info(model)

            try:
                C_SchedulerHook.INFERENCE_PREDICTOR = (
                    ConfigManager.get_inference_time_predictor(model, hw, sched_config)
                )
            except Exception as e:
                if disagg_cfg.enabled:
                    # Extend/decode batches are priced exclusively by the PD
                    # backend. An unavailable aggregate predictor is harmless.
                    C_SchedulerHook.INFERENCE_PREDICTOR = None
                    logger.warning(
                        "Failed to initialize global inference predictor (%s); "
                        "PD role predictors remain authoritative.",
                        e,
                    )
                else:
                    logger.error(
                        f"Failed to initialize inference time predictor. Error: {e}"
                    )
                    raise e

            # Phase 2b.3 / 5c.1 / 5c.2: optionally build the PD disagg backend.
            # Dispatches on disagg.backend ("single_process" → BackendA,
            # "two_process" → BackendB). start_pd_backend handles BackendB's
            # start() + atexit shutdown so workers can't leak.
            try:
                if disagg_cfg.enabled:
                    from hisim.simulation.pd_runtime import (
                        build_pd_backend,
                        start_pd_backend,
                    )

                    backend = build_pd_backend(
                        model=model,
                        base_sched_config=sched_config,
                        disagg_config=disagg_cfg,
                    )
                    C_SchedulerHook.PD_BACKEND = start_pd_backend(backend)
                    # Fresh run: clear every rid-keyed dict tied to the
                    # previous backend instance, not just the clocks.
                    # PD_REQUEST_STATES/REQUEST_STATS used to be left for
                    # wrapped_profile() to clear, but that only runs on a
                    # clean /profile_start flush -- a mid-run crash or an
                    # in-process Scheduler re-instantiation would otherwise
                    # leak stale (possibly terminal-phase) rid entries into
                    # this fresh run.
                    C_SchedulerHook.PD_REQUEST_STATES.clear()
                    C_SchedulerHook.REQUEST_STATS.clear()
                    C_SchedulerHook.PD_LAST_DECODE_STEP_END = 0.0
                    C_SchedulerHook.PD_PREFILL_KV_SERVICE.clear()
                    C_SchedulerHook.PD_CHUNK_ACCUM.clear()
                    logger.info(
                        "PD disaggregation enabled (backend=%s, prefill_replicas=%d, "
                        "decode_replicas=%d).",
                        disagg_cfg.backend,
                        C_SchedulerHook.PD_BACKEND.prefill_pool_size(),
                        C_SchedulerHook.PD_BACKEND.decode_pool_size(),
                    )
                else:
                    C_SchedulerHook.PD_BACKEND = None
                    C_SchedulerHook.PD_CLOSED_LOOP = False
            except Exception as e:
                logger.error(f"Failed to initialize PD backend. Error: {e}")
                raise e

        def wrapped_recv_requests(self, *args, **kwargs) -> list:
            recv_reqs = []

            if C_SchedulerHook.SIM_MODE == MockSimulationMode.BLOCKING:
                recv_reqs.extend(original_recv_requests(self, *args, **kwargs))
            elif C_SchedulerHook.SIM_MODE == MockSimulationMode.OFFLINE:
                # Initializing
                if not C_SchedulerHook.OFFLINE_RECV_ALL_REQUEST:
                    gen_requests = []
                    extra_requests = []
                    time.sleep(0.05)  # waiting requests

                    reqs = original_recv_requests(self, *args, **kwargs)

                    for req in reqs:
                        if req.__class__.__name__ == "TokenizedGenerateReqInput":
                            gen_requests.append(req)
                        else:
                            # Such as: /profile_start, /flush_cache, etc.
                            extra_requests.append(req)

                    # Add requests to future queue
                    for req in gen_requests:
                        sim_params = None
                        if req.sampling_params.custom_params is not None:
                            sim_params = req.sampling_params.custom_params.get(
                                "simulation"
                            )
                        if sim_params is None:
                            # There are some warm-up requests when starting the server without --skip-server-warmup.
                            extra_requests.append(req)
                            logger.warning(
                                "Failed to extract the simulation parameters required for simulation from the request. Ignore this warning if the request is a warm-up request."
                            )
                            continue
                        if sim_params.get("queue_start"):
                            logger.debug(
                                "Add request to waiting queue with custom queue start timestamp."
                            )

                        C_SchedulerHook.FUTURE_QUEUE.append(
                            (
                                sim_params.get("queue_start")
                                or sim_params["created_time"],
                                time.time_ns(),  # The request is not comparable, so add the salt to avoid comparison.
                                req,
                            )
                        )

                    if len(C_SchedulerHook.FUTURE_QUEUE) != 0:
                        _, _, gen_req = C_SchedulerHook.FUTURE_QUEUE[-1]
                        total_request = gen_req.sampling_params.custom_params[
                            "simulation"
                        ]["total_request"]

                        if len(C_SchedulerHook.FUTURE_QUEUE) == total_request:
                            C_SchedulerHook.OFFLINE_RECV_ALL_REQUEST = True
                            heapq.heapify(C_SchedulerHook.FUTURE_QUEUE)
                            logger.info(
                                "All requests received. Starting simulation now."
                            )
                        else:
                            logger.info(
                                f"Offline simulation mode enabled. {total_request} requests expected in total. Received {len(C_SchedulerHook.FUTURE_QUEUE)} requests so far."
                            )

                    if len(extra_requests) != 0:
                        # Schedule the extra requests immediately.
                        return extra_requests
                else:
                    # Extra requests include: flush request, abort request, etc.
                    recv_reqs.extend(original_recv_requests(self, *args, **kwargs))

                # Process the arrived requests only after all requests have been added to the future queue
                current_timestamp = StateManager.get_global_clock()
                while (
                    C_SchedulerHook.OFFLINE_RECV_ALL_REQUEST
                    and len(C_SchedulerHook.FUTURE_QUEUE) > 0
                ):
                    enqueue_time, _, req = C_SchedulerHook.FUTURE_QUEUE[0]
                    if enqueue_time > current_timestamp:
                        break
                    recv_reqs.append(req)
                    heapq.heappop(C_SchedulerHook.FUTURE_QUEUE)

            now = time.time()
            if C_SchedulerHook.PD_BACKEND is not None:
                from hisim.simulation.pd_metrics import (
                    flush_finished_states,
                    populate_request_stats,
                )
                from hisim.simulation.pd_sglang_lifecycle import abort_request_ids

                aborted_rids = set()
                for message in recv_reqs:
                    aborted_rids.update(abort_request_ids(message))
                if aborted_rids:
                    termination_time = (
                        now
                        if C_SchedulerHook.SIM_MODE == MockSimulationMode.BLOCKING
                        else StateManager.get_global_clock()
                    )
                    for rid in aborted_rids:
                        state = C_SchedulerHook.PD_REQUEST_STATES.get(rid)
                        if state is not None:
                            state.output_length = state.decode_step_count
                            C_SchedulerHook.PD_BACKEND.terminate_request(
                                state, termination_time
                            )
                            stats = C_SchedulerHook.REQUEST_STATS.get(rid)
                            if stats is not None:
                                stats.output_length = state.decode_step_count
                                populate_request_stats(stats, state)
                        C_SchedulerHook.PD_PREFILL_KV_SERVICE.pop(rid, None)
                        C_SchedulerHook.PD_CHUNK_ACCUM.pop(rid, None)
                    flush_finished_states(
                        C_SchedulerHook.PD_REQUEST_STATES,
                        C_SchedulerHook.REQUEST_STATS,
                    )
                    if C_SchedulerHook.FUTURE_QUEUE:
                        C_SchedulerHook.FUTURE_QUEUE = [
                            item
                            for item in C_SchedulerHook.FUTURE_QUEUE
                            if getattr(item[2], "rid", None) not in aborted_rids
                        ]
                        heapq.heapify(C_SchedulerHook.FUTURE_QUEUE)
            for req in recv_reqs:
                if req.__class__.__name__ in [
                    "BatchTokenizedGenerateReqInput",
                    "TokenizedGenerateReqInput",
                ]:
                    req_stats = C_SchedulerHook.REQUEST_STATS[req.rid]
                    req_stats.rid = req.rid
                    req_stats.input_length = len(req.input_ids)
                    req_stats.output_length = req.sampling_params.max_new_tokens
                    simulation_args = req.sampling_params.custom_params["simulation"]
                    if C_SchedulerHook.SIM_MODE == MockSimulationMode.BLOCKING:
                        if "server_created_time" not in simulation_args:
                            logger.warning(
                                "The request's creation time is missing, which may cause the TTFT to be inaccurate."
                            )
                        req_stats.created_time = simulation_args.get(
                            "server_created_time", now
                        )
                        req_stats.last_event_time = req_stats.created_time
                        req_stats.queue_start = now
                    elif C_SchedulerHook.SIM_MODE == MockSimulationMode.OFFLINE:
                        req_stats.created_time = simulation_args["created_time"]
                        req_stats.last_event_time = req_stats.created_time
                        # Align with the real queue start timestamp if queue_start is not None. For debugging only.
                        queue_start = simulation_args.get("queue_start")
                        if queue_start is not None:
                            StateManager.set_global_clock(queue_start)
                        req_stats.queue_start = StateManager.get_global_clock()

            if recv_reqs and C_SchedulerHook.LAST_CPU_TS == 0:
                C_SchedulerHook.LAST_CPU_TS = time.time()
                C_SchedulerHook.LAST_FLUSH_TS = C_SchedulerHook.LAST_CPU_TS
                StateManager.set_global_clock(0)
                # Anchor PD_LAST_DECODE_STEP_END to the same t=0 origin as
                # the global clock at the start of a run.
                C_SchedulerHook.PD_LAST_DECODE_STEP_END = 0.0
                C_SchedulerHook.PD_PREFILL_KV_SERVICE.clear()
                C_SchedulerHook.PD_CHUNK_ACCUM.clear()

            return recv_reqs

        def wrapped_get_new_batch_prefill(self, *args, **kwargs):
            new_batch = original_get_new_batch_prefill(self, *args, **kwargs)
            now = time.time()
            if new_batch is not None:
                for req in new_batch.reqs:
                    req_stats = C_SchedulerHook.REQUEST_STATS[req.rid]
                    req_stats.final_reused_tokens = req.cached_tokens
                    if req_stats.queue_end == -1:
                        if C_SchedulerHook.SIM_MODE == MockSimulationMode.BLOCKING:
                            req_stats.queue_end = now
                        else:
                            req_stats.queue_end = StateManager.get_global_clock()
                    else:
                        # Chunked request
                        pass
            elif len(self.running_batch.reqs) == 0 and len(self.waiting_queue) > 0:
                # Prefetching
                StateManager.step_global_clock(0.005)
                StateManager.set_current_inference_dur(0.005)
            else:
                if C_SchedulerHook.SIM_MODE == MockSimulationMode.OFFLINE and (
                    len(C_SchedulerHook.FUTURE_QUEUE) != 0
                    and len(self.running_batch.reqs) == 0
                ):
                    next_created_time, _, req = C_SchedulerHook.FUTURE_QUEUE[0]
                    StateManager.set_global_clock(next_created_time + 1e-6)
            logger.debug(
                f"Get new batch prefill: global iteration={StateManager.get_iteration()}, "
                f"new batch={new_batch.batch_size() if new_batch is not None else 0}, "
                f"waiting queue={len(self.waiting_queue)}"
            )

            return new_batch

        def wrapped_update_running_batch(self, batch, *args, **kwargs):
            # SGLang's KV-cache-pressure retraction (ScheduleBatch.retract_decode)
            # runs *inside* the original call below: it evicts a running-decode
            # request's KV cache and re-queues the same rid (stamping
            # `is_retracted=True`) so it re-enters prefill and rebuilds KV over
            # (original prompt + already-generated output). HiSim's
            # PD_REQUEST_STATES entry for that rid is still parked wherever its
            # own (decoupled) virtual PD clock left it -- KV_TRANSIT,
            # WAITING_DECODE, or RUNNING_DECODE are all possible -- and must be
            # snapped back to WAITING_PREFILL here, before the *next* scheduler
            # iteration's get_new_batch_prefill call can pick the retracted
            # request back up. (get_new_batch_prefill always runs before
            # update_running_batch within the same iteration -- see
            # get_next_batch_to_run's call order -- so a retraction detected
            # here is guaranteed to land one full iteration ahead of the
            # re-prefill that would otherwise crash on a stale RUNNING_DECODE
            # phase: "cannot schedule prefill for rid=... in phase=running_decode".)
            pre_rids = (
                {req.rid for req in batch.reqs}
                if batch is not None and getattr(batch, "reqs", None)
                else set()
            )
            result = original_update_running_batch(self, batch, *args, **kwargs)

            if pre_rids and C_SchedulerHook.PD_BACKEND is not None:
                from hisim.simulation.pd_sglang_lifecycle import (
                    retracted_request_ids,
                )

                retracted_rids = retracted_request_ids(self.waiting_queue, pre_rids)
                if retracted_rids:
                    now = (
                        time.time()
                        if C_SchedulerHook.SIM_MODE == MockSimulationMode.BLOCKING
                        else StateManager.get_global_clock()
                    )
                    logger.info(
                        "SGLang retracted %d request(s) for KV-cache pressure; "
                        "resetting PD state to WAITING_PREFILL: %s",
                        len(retracted_rids),
                        sorted(retracted_rids),
                    )
                    for rid in retracted_rids:
                        state = C_SchedulerHook.PD_REQUEST_STATES.get(rid)
                        if state is not None:
                            C_SchedulerHook.PD_BACKEND.reset_for_retract(state, now)
                        C_SchedulerHook.PD_CHUNK_ACCUM.pop(rid, None)
                        C_SchedulerHook.PD_PREFILL_KV_SERVICE.pop(rid, None)

            return result

        def wrapped_run_batch(self, *args, **kwargs):
            ret = original_run_batch(self, *args, **kwargs)

            batch = get_obj_from_args(
                "sglang.srt.managers.schedule_batch.ScheduleBatch", *args, **kwargs
            )
            if batch is None:
                for candidate in list(args) + list(kwargs.values()):
                    if hasattr(candidate, "reqs") and hasattr(candidate, "forward_mode"):
                        batch = candidate
                        break

            # In recent SGLang multiprocessing layouts, run_batch can execute in
            # a worker process that does not carry over class-level PD backend
            # state from wrapped_init. Build it lazily on first use.
            if C_SchedulerHook.PD_BACKEND is None:
                disagg_cfg = ConfigManager.get_disagg_config()
                if disagg_cfg.enabled:
                    cfg_closed_loop = False
                    try:
                        with open(Envs.config_path(), "r", encoding="utf-8") as f:
                            cfg_closed_loop = bool(
                                json.load(f)
                                .get("disagg", {})
                                .get("closed_loop_ttft", False)
                            )
                    except Exception:
                        cfg_closed_loop = False
                    C_SchedulerHook.PD_CLOSED_LOOP = (
                        cfg_closed_loop or Envs.pd_closed_loop()
                    )
                    from hisim.simulation.pd_runtime import (
                        build_pd_backend,
                        start_pd_backend,
                    )

                    model = ConfigManager.get_model_info(
                        self.model_config.hf_config.__dict__
                    )
                    sched_config = ConfigManager.get_scheduler_config(
                        self.server_args.__dict__,
                        "sglang",
                        self.model_config.hf_config.__dict__,
                    )
                    backend = build_pd_backend(
                        model=model,
                        base_sched_config=sched_config,
                        disagg_config=disagg_cfg,
                    )
                    C_SchedulerHook.PD_BACKEND = start_pd_backend(backend)
                    C_SchedulerHook.PD_REQUEST_STATES.clear()
                    C_SchedulerHook.REQUEST_STATS.clear()
                    C_SchedulerHook.PD_LAST_DECODE_STEP_END = 0.0
                    C_SchedulerHook.PD_PREFILL_KV_SERVICE.clear()
                    C_SchedulerHook.PD_CHUNK_ACCUM.clear()
                    logger.info(
                        "PD backend lazily initialized in run_batch (backend=%s, closed_loop=%s).",
                        disagg_cfg.backend,
                        C_SchedulerHook.PD_CLOSED_LOOP,
                    )

            if batch is not None and hasattr(batch, "forward_mode"):
                hisim_batch = HisimScheduleBatch(reqs=[])
                C_SchedulerHook.PD_BATCH_TOKEN_TIMES = {}
                if batch.forward_mode.is_extend():
                    for req in batch.reqs:
                        hisim_batch.reqs.append(
                            FakeRequest(
                                input_length=req.extend_input_len,
                                past_kv_length=len(req.prefix_indices)
                                + len(req.output_ids),
                            )
                        )
                elif batch.forward_mode.is_decode():
                    for req in batch.reqs:
                        hisim_batch.reqs.append(
                            FakeRequest(
                                input_length=1,
                                past_kv_length=len(req.prefix_indices)
                                + len(req.output_ids),
                            )
                        )

                if not hisim_batch.is_empty():
                    StateManager.inc_iteration()
                    pd_priced_batch = (
                        C_SchedulerHook.PD_BACKEND is not None
                        and (
                            batch.forward_mode.is_extend()
                            or batch.forward_mode.is_decode()
                        )
                    )
                    if pd_priced_batch:
                        # The role-specific backend below is authoritative.
                        # Avoid a redundant aggregate predictor call whose result
                        # would be discarded immediately.
                        predicted_latency = 0.0
                    else:
                        predicted_latency = float(
                            C_SchedulerHook.INFERENCE_PREDICTOR.predict_infer_time(
                                hisim_batch
                            )
                        )

                    # Phase 2b.4a: when PD is active, override prefill latency
                    # with the BackendA prefill pool clock. Decode path still
                    # uses the aggregated predictor (covered in 2b.4b).
                    if (
                        C_SchedulerHook.PD_BACKEND is not None
                        and batch.forward_mode.is_extend()
                    ):
                        from hisim.simulation.pd_runtime import (
                            admit_prefill_batch_latency,
                            finalize_prefill_batch,
                            record_prefill_sampled_tokens,
                        )
                        from hisim.simulation.pd_types import (
                            PDRequestState,
                            RequestPhase,
                        )
                        from hisim.simulation.pd_metrics import (
                            populate_request_stats,
                        )
                        from hisim.simulation.pd_timeline import (
                            prefill_admission_baseline,
                            prefill_batch_start,
                            prefill_queue_baseline,
                        )

                        # P1: prefill role clock. The batch starts no earlier
                        # than the prefill engine is free AND every request has
                        # been admitted by SGLang's native waiting queue. Using
                        # queue_end (with created_time fallback) prevents the PD
                        # role clock from retroactively erasing queue wait.
                        #
                        # "Prefill engine is free" is read straight off the
                        # backend's own replica pool (min busy_until across all
                        # prefill replicas), NOT a hand-tracked scalar: with
                        # prefill_replicas > 1, a scalar would only remember the
                        # one replica the previous extend batch happened to land
                        # on, and would force this batch to wait for that
                        # specific replica even when a different one has been
                        # idle the whole time.
                        admission_times = []
                        for req in batch.reqs:
                            st = C_SchedulerHook.REQUEST_STATS.get(req.rid)
                            admission_times.append(
                                prefill_admission_baseline(
                                    st.created_time
                                    if st is not None
                                    else 0.0,
                                    st.queue_end if st is not None else None,
                                )
                            )
                        now_clock = prefill_batch_start(
                            C_SchedulerHook.PD_BACKEND.earliest_pool_time(
                                "prefill"
                            ),
                            admission_times,
                        )

                        # Build per-request states. Accumulate chunk token counts
                        # so full prompt length is available at finalize time.
                        # Mid-chunk requests (already in RUNNING_PREFILL) reuse
                        # their existing state; the backend's phase guard ensures
                        # on_request_arrival / admit_prefill are NOT re-run on
                        # them, preventing phase reset and double-counted time.
                        states = []
                        for req in batch.reqs:
                            chunk_len = int(req.extend_input_len)
                            # Running sum across chunks for full-prompt fallback.
                            C_SchedulerHook.PD_CHUNK_ACCUM[req.rid] = (
                                C_SchedulerHook.PD_CHUNK_ACCUM.get(req.rid, 0)
                                + chunk_len
                            )
                            s = C_SchedulerHook.PD_REQUEST_STATES.get(req.rid)
                            if s is None:
                                # First (or only) chunk: create state.
                                st = C_SchedulerHook.REQUEST_STATS.get(req.rid)
                                arrival = now_clock
                                if (
                                    st is not None
                                    and st.created_time is not None
                                    and st.created_time >= 0.0
                                ):
                                    arrival = st.created_time
                                s = PDRequestState(
                                    rid=req.rid,
                                    arrival_time=arrival,
                                    prefill_queue_start_time=prefill_queue_baseline(
                                        arrival,
                                        st.queue_start if st is not None else None,
                                    ),
                                    phase=RequestPhase.WAITING_PREFILL,
                                    input_length=chunk_len,
                                    output_length=int(
                                        getattr(
                                            req.sampling_params,
                                            "max_new_tokens",
                                            0,
                                        )
                                        or 0
                                    ),
                                )
                                C_SchedulerHook.PD_REQUEST_STATES[req.rid] = s
                            else:
                                # Continuing chunk: update input_length to this
                                # chunk's token count for per-chunk latency
                                # accuracy (one forward pass ≈ chunk_len tokens).
                                s.input_length = chunk_len
                            s.prefill_is_final_chunk = (
                                getattr(req, "is_chunked", 0) <= 0
                            )
                            states.append(s)

                        # Compute batch latency. The backend phase guard in
                        # try_admit_prefill_batch skips controller transitions
                        # for any state already past WAITING_PREFILL, so only
                        # first-chunk states go through on_request_arrival +
                        # admit_prefill.
                        pd_latency = admit_prefill_batch_latency(
                            C_SchedulerHook.PD_BACKEND, states, now_clock
                        )

                        # Finalize only requests at their final (or only) chunk.
                        # is_chunked <= 0: this extend batch completes the
                        # request's prefill phase.  is_chunked != 0: more chunks
                        # follow, so leave the request in RUNNING_PREFILL.
                        # Before finalizing, set input_length to the full
                        # rebuilt-KV length so KV transfer sizing and decode KV
                        # base are accurate.
                        final_states = []
                        for req, s in zip(batch.reqs, states):
                            if getattr(req, "is_chunked", 0) <= 0:
                                fill_ids = getattr(req, "fill_ids", None)
                                if fill_ids:
                                    # fill_ids = origin_input_ids + output_ids.
                                    # For a request's first-ever prefill,
                                    # output_ids is always empty here, so this
                                    # is numerically identical to the
                                    # origin_input_ids-only value below. For a
                                    # retracted request's re-prefill, SGLang
                                    # rebuilds KV over the original prompt AND
                                    # every token already generated before the
                                    # retraction, so origin_input_ids alone
                                    # would silently under-count the true
                                    # rebuilt KV length by however many tokens
                                    # had already been decoded.
                                    s.input_length = len(fill_ids)
                                elif (
                                    hasattr(req, "origin_input_ids")
                                    and req.origin_input_ids
                                ):
                                    s.input_length = len(req.origin_input_ids)
                                else:
                                    # Fallback: use accumulated chunk sum.
                                    s.input_length = (
                                        C_SchedulerHook.PD_CHUNK_ACCUM.get(
                                            req.rid, s.input_length
                                        )
                                    )
                                C_SchedulerHook.PD_CHUNK_ACCUM.pop(req.rid, None)
                                final_states.append(s)

                        # Transfer final-prefill logits/KV to the selected D
                        # instance, then sample the first token there. Sampling
                        # has no full decode-forward cost in the current model.
                        if final_states:
                            finalize_prefill_batch(
                                C_SchedulerHook.PD_BACKEND,
                                final_states,
                                now_clock + pd_latency,
                            )
                            first_token_times = record_prefill_sampled_tokens(
                                C_SchedulerHook.PD_BACKEND,
                                final_states,
                            )
                            C_SchedulerHook.PD_BATCH_TOKEN_TIMES.update(
                                first_token_times
                            )
                            if first_token_times:
                                C_SchedulerHook.PD_LAST_DECODE_STEP_END = max(
                                    first_token_times.values()
                                )

                        # Record prefill+KV service time (closed-loop TTFT) and
                        # populate diagnostic stats for finalized requests only.
                        for s in final_states:
                            if s.kv_ready_time is not None:
                                if s.prefill_start_time is not None:
                                    prefill_kv_service = max(
                                        s.kv_ready_time
                                        - s.prefill_start_time,
                                        0.0,
                                    )
                                else:
                                    # Some SGLang paths do not stamp
                                    # prefill_start_time onto PDRequestState.
                                    # Reconstruct prefill+KV service from the
                                    # observed batch window plus kv_ready.
                                    prefill_end = (
                                        s.prefill_end_time
                                        if s.prefill_end_time is not None
                                        else now_clock + pd_latency
                                    )
                                    prefill_kv_service = max(
                                        prefill_end - now_clock, 0.0
                                    ) + max(
                                        s.kv_ready_time - prefill_end,
                                        0.0,
                                    )
                                C_SchedulerHook.PD_PREFILL_KV_SERVICE[s.rid] = (
                                    prefill_kv_service
                                )
                            stats = C_SchedulerHook.REQUEST_STATS.get(s.rid)
                            if stats is not None:
                                populate_request_stats(stats, s)

                        # P1: KV transfer time is NOT added to any advancing
                        # clock (fixes #2): it is encoded solely in per-request
                        # kv_ready_time (set by finalize_prefill_batch above),
                        # and the decode role clock gates on it via
                        # sync_decode_start. This stops a prefill batch + its KV
                        # transfer from freezing in-flight decode in virtual
                        # time. The prefill engine's own "free at" state lives
                        # entirely on PD_BACKEND's replica pool (busy_until,
                        # updated inside try_admit_prefill_batch above) -- there
                        # is no separate scalar to advance here.
                        logger.debug(
                            "[PD] extend batch: %d reqs (%d final-chunk), "
                            "agg_pred=%.6fs, pd_pred=%.6fs, prefill_batch_end=%.6fs",
                            len(states),
                            len(final_states),
                            predicted_latency,
                            pd_latency,
                            now_clock + pd_latency,
                        )
                        predicted_latency = pd_latency
                    elif (
                        C_SchedulerHook.PD_BACKEND is not None
                        and batch.forward_mode.is_decode()
                    ):
                        from hisim.simulation.pd_runtime import (
                            decode_batch_latency,
                        )
                        from hisim.simulation.pd_types import RequestPhase
                        from hisim.simulation.pd_metrics import (
                            populate_request_stats,
                        )
                        from hisim.simulation.pd_timeline import (
                            sync_decode_start,
                        )

                        # P1: decode role clock. The decode engine is free at
                        # PD_BACKEND.earliest_pool_time("decode") -- the backend
                        # decides internally whether that means the pool minimum
                        # (decode_queue_mode="per_replica_queue": replicas hold
                        # disjoint, independently-progressing requests, so a busy
                        # replica must never hold back a different, idle one) or
                        # the pool maximum (the default single_replica mode /
                        # BackendB: every step bundles the whole cohort onto one
                        # freshly-chosen replica, so the next step must wait for
                        # the last one's actual completion regardless of which
                        # replica handled it). The step also cannot begin before
                        # the KV cache of any request joining this batch has
                        # arrived. sync_decode_start jumps the clock forward to
                        # the latest such kv_ready_time (the prefill->decode
                        # handoff), and is otherwise driven purely by decode step
                        # latency — so prefill/KV work never freezes in-flight
                        # decode.
                        ctrl = C_SchedulerHook.PD_BACKEND.controller()
                        if (
                            C_SchedulerHook.PD_BACKEND.decode_queue_mode()
                            == "per_replica_queue"
                        ):
                            batch_states = [
                                C_SchedulerHook.PD_REQUEST_STATES[req.rid]
                                for req in batch.reqs
                                if req.rid in C_SchedulerHook.PD_REQUEST_STATES
                            ]
                            replica_by_rid = (
                                C_SchedulerHook.PD_BACKEND.bind_decode_replicas(
                                    batch_states
                                )
                            )
                            bucket_rids: dict[int, list[str]] = {}
                            for req in batch.reqs:
                                state = C_SchedulerHook.PD_REQUEST_STATES.get(
                                    req.rid
                                )
                                if state is None:
                                    continue
                                replica_idx = replica_by_rid[state.rid]
                                bucket_rids.setdefault(replica_idx, []).append(
                                    state.rid
                                )

                            # Compute every bucket's own synced step_start
                            # BEFORE polling. Each bucket's floor starts at
                            # that replica's own busy_until and is then
                            # advanced (sync_decode_start only ever moves
                            # forward) past the kv_ready_time of its own
                            # in-flight requests -- so one bucket's floor can
                            # end up strictly later than another's.
                            #
                            # Bug fix: polling with only the MINIMUM raw
                            # replica clock (as this used to do) under-
                            # promotes: any request whose kv_ready_time falls
                            # between that minimum and its OWN bucket's later
                            # synced step_start never leaves KV_TRANSIT, so
                            # admit_decode_for_replica can't admit it even
                            # though its own bucket's step_start has genuinely
                            # reached its kv_ready_time. When this happens to
                            # every bucket in the round, token_times stays
                            # empty and the "no admissible request state"
                            # RuntimeError below fires spuriously -- a real
                            # bug that only surfaces with 2+ decode replicas
                            # whose busy_until/kv_ready_time straddle that
                            # minimum (e.g. live 1P2D traffic).
                            #
                            # Polling once with the MAXIMUM of the buckets'
                            # floors instead of the minimum keeps this a
                            # single O(K) scan of the shared _kv_transit list
                            # (not O(N*K), one scan per bucket) while still
                            # promoting every request each bucket will
                            # actually need this round: poll_kv_ready is
                            # monotonic, so polling once with the max is
                            # equivalent to polling per bucket in increasing
                            # order.
                            bucket_step_start: dict[int, float] = {}
                            for replica_idx, rids in bucket_rids.items():
                                step_start = (
                                    C_SchedulerHook.PD_BACKEND.decode_replica_time(
                                        replica_idx
                                    )
                                )
                                for rid in rids:
                                    s = C_SchedulerHook.PD_REQUEST_STATES[rid]
                                    if s.phase in (
                                        RequestPhase.KV_TRANSIT,
                                        RequestPhase.WAITING_DECODE,
                                    ):
                                        step_start = sync_decode_start(
                                            step_start, s.kv_ready_time
                                        )
                                bucket_step_start[replica_idx] = step_start

                            ctrl.poll_kv_ready(max(bucket_step_start.values()))

                            token_times: dict[str, float] = {}
                            bucket_step_starts = []
                            bucket_step_ends = []
                            running_count = 0
                            for replica_idx in sorted(
                                bucket_rids,
                                key=C_SchedulerHook.PD_BACKEND.decode_replica_time,
                            ):
                                step_start = bucket_step_start[replica_idx]
                                # Only offer requests that actually need
                                # (re-)admission this round. admit_decode_for_
                                # replica un-binds any rid it doesn't admit --
                                # correct for a genuine capacity rejection, but
                                # a request already RUNNING_DECODE (continuing
                                # from a prior round, bundled into this same
                                # native batch) was never a candidate for
                                # admission in the first place. Including it
                                # here would spuriously strip its sticky
                                # replica binding every single round, letting
                                # bind_decode_replicas silently reassign it to
                                # a different replica next time -- breaking the
                                # very stickiness this mode exists to provide.
                                pending_rids = {
                                    rid
                                    for rid in bucket_rids[replica_idx]
                                    if C_SchedulerHook.PD_REQUEST_STATES[rid].phase
                                    != RequestPhase.RUNNING_DECODE
                                }
                                if pending_rids:
                                    C_SchedulerHook.PD_BACKEND.admit_decode_for_replica(
                                        replica_idx,
                                        pending_rids,
                                        step_start,
                                    )
                                states = [
                                    C_SchedulerHook.PD_REQUEST_STATES[rid]
                                    for rid in bucket_rids[replica_idx]
                                    if C_SchedulerHook.PD_REQUEST_STATES[rid].phase
                                    == RequestPhase.RUNNING_DECODE
                                ]
                                if not states:
                                    continue
                                pd_latency = decode_batch_latency(
                                    C_SchedulerHook.PD_BACKEND,
                                    states,
                                    step_start,
                                )
                                step_end = step_start + pd_latency
                                bucket_step_starts.append(step_start)
                                bucket_step_ends.append(step_end)
                                running_count += len(states)
                                for s in states:
                                    token_times[s.rid] = step_end
                                # Bookkeeping: credit one decode step per request.
                                C_SchedulerHook.PD_BACKEND.on_decode_step_done_batch(
                                    states, step_end
                                )
                                for s in states:
                                    stats = C_SchedulerHook.REQUEST_STATS.get(
                                        s.rid
                                    )
                                    if stats is not None:
                                        populate_request_stats(stats, s)
                            if token_times:
                                # P1: record tokens on each replica's own decode
                                # timeline, not a synthetic batch-wide end time.
                                C_SchedulerHook.PD_BATCH_TOKEN_TIMES = token_times
                                C_SchedulerHook.PD_LAST_DECODE_STEP_END = max(
                                    bucket_step_ends
                                )
                                # Bug fix: report the round's real-time cost as
                                # the SLOWEST bucket's OWN step latency (its own
                                # step_end - step_start), not the span from the
                                # earliest bucket's start to the latest bucket's
                                # end. Buckets are deliberately independent replica
                                # clocks (see decode_replica_time / the "own
                                # busy_until, not the slowest" regression test) --
                                # a bucket that finished a lighter batch earlier is
                                # correctly idle, not "behind"; the previous
                                # max(ends) - min(starts) formula mistook that
                                # idle gap for extra round latency and re-charged
                                # it (via time.sleep) every single round. Because a
                                # request can take up to output_len decode rounds,
                                # even a small per-round overcount compounds into
                                # multi-second inflation as decode replica count
                                # (and therefore the chance of an idle/busy split)
                                # grows -- exactly the E2E/TTFT blowup observed
                                # empirically in 1P2D..1P16D live-server runs.
                                round_latency = max(
                                    end - start
                                    for start, end in zip(
                                        bucket_step_starts, bucket_step_ends
                                    )
                                )
                                logger.debug(
                                    "[PD] decode batch: %d reqs across %d decode replicas, "
                                    "agg_pred=%.6fs, pd_pred=%.6fs, decode_clock=%.6fs",
                                    running_count,
                                    len(bucket_step_ends),
                                    predicted_latency,
                                    round_latency,
                                    C_SchedulerHook.PD_LAST_DECODE_STEP_END,
                                )
                                predicted_latency = round_latency
                            elif batch.reqs:
                                raise RuntimeError(
                                    "PD decode batch contained no admissible "
                                    "request state; native and PD capacity/state "
                                    "tracking diverged"
                                )
                        else:
                            step_start = (
                                C_SchedulerHook.PD_BACKEND.earliest_pool_time(
                                    "decode"
                                )
                            )
                            for req in batch.reqs:
                                s = C_SchedulerHook.PD_REQUEST_STATES.get(req.rid)
                                if (
                                    s is not None
                                    and s.phase
                                    in (
                                        RequestPhase.KV_TRANSIT,
                                        RequestPhase.WAITING_DECODE,
                                    )
                                ):
                                    step_start = sync_decode_start(
                                        step_start, s.kv_ready_time
                                    )
                            ctrl.poll_kv_ready(step_start)
                            C_SchedulerHook.PD_BACKEND.admit_decode_single_replica(
                                {
                                    req.rid
                                    for req in batch.reqs
                                    if req.rid in C_SchedulerHook.PD_REQUEST_STATES
                                },
                                step_start,
                            )
                            # Only include requests that are both in the current
                            # SGLang decode batch and have been admitted onto the
                            # PD decode pool (phase == RUNNING_DECODE). The phase
                            # filter prevents KV_TRANSIT requests from being decoded
                            # prematurely, while targeted admission above avoids
                            # stamping decode_start_time on unrelated waiters that
                            # are not part of this concrete batch.
                            states = [
                                C_SchedulerHook.PD_REQUEST_STATES[req.rid]
                                for req in batch.reqs
                                if req.rid in C_SchedulerHook.PD_REQUEST_STATES
                                and C_SchedulerHook.PD_REQUEST_STATES[req.rid].phase
                                == RequestPhase.RUNNING_DECODE
                            ]
                            if states:
                                pd_latency = decode_batch_latency(
                                    C_SchedulerHook.PD_BACKEND, states, step_start
                                )
                                step_end = step_start + pd_latency
                                C_SchedulerHook.PD_BATCH_TOKEN_TIMES = {
                                    s.rid: step_end for s in states
                                }
                                # Bookkeeping: credit one decode step per request.
                                C_SchedulerHook.PD_BACKEND.on_decode_step_done_batch(
                                    states, step_end
                                )
                                for s in states:
                                    stats = C_SchedulerHook.REQUEST_STATS.get(s.rid)
                                    if stats is not None:
                                        populate_request_stats(stats, s)
                                # P1: remember the step end so process_batch_result
                                # records this batch's tokens on the decode timeline
                                # (not the prefill/KV-polluted global clock). The
                                # decode engine's own "free at" state lives entirely
                                # on PD_BACKEND's replica pool (busy_until, updated
                                # inside try_admit_decode_batch above) -- there is no
                                # separate scalar to advance here.
                                C_SchedulerHook.PD_LAST_DECODE_STEP_END = step_end
                                logger.debug(
                                    "[PD] decode batch: %d reqs, agg_pred=%.6fs, "
                                    "pd_pred=%.6fs, decode_clock=%.6fs",
                                    len(states),
                                    predicted_latency,
                                    pd_latency,
                                    step_end,
                                )
                                predicted_latency = pd_latency
                            else:
                                raise RuntimeError(
                                    "PD decode batch contained no admissible "
                                    "request state; native and PD capacity/state "
                                    "tracking diverged"
                                )

                    forward_latency = 0
                    if C_SchedulerHook.SIM_MODE == MockSimulationMode.BLOCKING:
                        now = time.time()
                        time.sleep(abs(predicted_latency))
                        now = time.time()
                        forward_latency = now - C_SchedulerHook.LAST_CPU_TS
                        C_SchedulerHook.LAST_CPU_TS = now
                    else:
                        now = time.time()
                        forward_latency = predicted_latency

                    StateManager.set_current_inference_dur(forward_latency)

                C_SchedulerHook.HISIM_BATCH = hisim_batch

            return ret

        def wrapped_process_batch_result(self, *args, **kwargs):
            batch = get_obj_from_args(
                "sglang.srt.managers.schedule_batch.ScheduleBatch", *args, **kwargs
            )
            batch_reqs = list(batch.reqs) if batch is not None else []
            ret = original_process_batch_result(self, *args, **kwargs)

            if batch is not None:
                if not batch_reqs:
                    return ret

                hicache_l2_load_dur = StateManager.pop_hicache_l2_load_dur()
                hicache_l2_backup_dur = StateManager.pop_hicache_l2_backup_dur()
                current_inference_dur = StateManager.get_current_inference_dur()

                if C_SchedulerHook.OVERLAP_SCHEDULE:
                    StateManager.step_global_clock(
                        max(
                            hicache_l2_load_dur - StateManager.get_last_inference_dur(),
                            0,
                        )
                    )
                    StateManager.step_global_clock(current_inference_dur)
                    request_response_time = (
                        StateManager.get_global_clock() + hicache_l2_backup_dur
                    )
                else:
                    StateManager.step_global_clock(
                        hicache_l2_load_dur
                        + current_inference_dur
                        + hicache_l2_backup_dur
                    )
                    request_response_time = StateManager.get_global_clock()
                pd_role_token_mode = (
                    C_SchedulerHook.PD_BACKEND is not None
                    and hasattr(batch, "forward_mode")
                    and (
                        batch.forward_mode.is_extend()
                        or batch.forward_mode.is_decode()
                    )
                )
                pd_closed_loop = (
                    pd_role_token_mode and C_SchedulerHook.PD_CLOSED_LOOP
                )
                if pd_closed_loop:
                    from hisim.simulation.pd_timeline import (
                        closed_loop_first_token_latency,
                    )

                # Request statistics
                for req in batch_reqs:
                    if req.is_chunked <= 0:
                        req_stats = C_SchedulerHook.REQUEST_STATS[req.rid]
                        # SGLang samples one token while processing the final
                        # extend result.  PD prices that token on the D pool;
                        # PD_BATCH_TOKEN_TIMES therefore decides whether this
                        # concrete extend/decode result contributes a token.
                        if (
                            C_SchedulerHook.PD_BACKEND is not None
                            and hasattr(batch, "forward_mode")
                            and (
                                batch.forward_mode.is_extend()
                                or batch.forward_mode.is_decode()
                            )
                            and req.rid
                            not in C_SchedulerHook.PD_BATCH_TOKEN_TIMES
                        ):
                            pass  # this PD role batch did not emit a token for this rid
                        elif (
                            C_SchedulerHook.PD_BACKEND is not None
                            and hasattr(batch, "forward_mode")
                            and (
                                batch.forward_mode.is_extend()
                                or batch.forward_mode.is_decode()
                            )
                        ):
                            # P1: record the decode token on the decode role
                            # clock (PD_LAST_DECODE_STEP_END) instead of the
                            # global clock. For the first token last_event_time
                            # is still the arrival time, so gen[0] == full TTFT;
                            # for later tokens it is the previous step end, so
                            # gen[i] == decode step latency (ITL). This makes
                            # E2E = sum(gen) telescope to decode_end - arrival,
                            # and last_event_time track the true decode makespan
                            # — all free of prefill/KV serialisation.
                            token_time = C_SchedulerHook.PD_BATCH_TOKEN_TIMES.get(
                                req.rid,
                                C_SchedulerHook.PD_LAST_DECODE_STEP_END,
                            )
                            if pd_closed_loop and not req_stats.gen_token_latencies:
                                state = C_SchedulerHook.PD_REQUEST_STATES.get(
                                    req.rid
                                )
                                service_ttft = None
                                if state is not None:
                                    # Closed-loop emulation feeds all arrivals
                                    # at virtual t=0. Measure first-token TTFT
                                    # from the request's service spans to drop
                                    # the synthetic cap queueing.
                                    service_ttft = closed_loop_first_token_latency(
                                        token_time,
                                        prefill_start_time=state.prefill_start_time,
                                        kv_ready_time=state.kv_ready_time,
                                        decode_start_time=state.decode_start_time,
                                    )
                                if service_ttft is None:
                                    prefill_kv = (
                                        C_SchedulerHook.PD_PREFILL_KV_SERVICE.get(
                                            req.rid
                                        )
                                    )
                                    if prefill_kv is not None:
                                        service_ttft = (
                                            prefill_kv
                                            + max(current_inference_dur, 0.0)
                                        )
                                if service_ttft is not None:
                                    req_stats.gen_token_latencies.append(
                                        service_ttft
                                    )
                                else:
                                    req_stats.gen_token_latencies.append(
                                        token_time
                                        - req_stats.last_event_time
                                    )
                            else:
                                req_stats.gen_token_latencies.append(
                                    token_time - req_stats.last_event_time
                                )
                            req_stats.last_event_time = token_time
                        else:
                            req_stats.gen_token_latencies.append(
                                request_response_time
                                - req_stats.last_event_time
                            )
                            req_stats.last_event_time = request_response_time
                    else:
                        # Chunked request: nothing to do
                        pass
                if C_SchedulerHook.PD_BACKEND is not None:
                    # Flush only after token statistics have consumed the
                    # terminal state.  This is required for OSL=1, which
                    # finishes on the token sampled by final prefill.
                    from hisim.simulation.pd_metrics import flush_finished_states
                    from hisim.simulation.pd_types import RequestPhase

                    finished_rids = {
                        rid
                        for rid, state in C_SchedulerHook.PD_REQUEST_STATES.items()
                        if state.phase == RequestPhase.FINISHED
                    }
                    flushed = flush_finished_states(
                        C_SchedulerHook.PD_REQUEST_STATES,
                        C_SchedulerHook.REQUEST_STATS,
                    )
                    for rid in finished_rids:
                        C_SchedulerHook.PD_PREFILL_KV_SERVICE.pop(rid, None)
                        C_SchedulerHook.PD_CHUNK_ACCUM.pop(rid, None)
                    if flushed:
                        logger.debug("[PD] flushed %d finished states", flushed)
                # Iteration statistics
                C_SchedulerHook.ITERATION_STATS.append(
                    {
                        "requests": C_SchedulerHook.HISIM_BATCH.request_info(),
                        "forward_latency": current_inference_dur,
                        "l2_load_latency": hicache_l2_load_dur,
                        "l2_backup_latency": hicache_l2_backup_dur,
                    }
                )
                C_SchedulerHook.PD_BATCH_TOKEN_TIMES = {}
            C_SchedulerHook.LAST_CPU_TS = time.time()
            return ret

        def wrapped_profile(self, req, *args, **kwargs):
            # Final safety-net flush so any FINISHED PD states whose decode
            # tick fired after the last GC point still contribute stage
            # percentiles to the metrics dump.
            if C_SchedulerHook.PD_REQUEST_STATES:
                from hisim.simulation.pd_metrics import flush_finished_states

                flush_finished_states(
                    C_SchedulerHook.PD_REQUEST_STATES,
                    C_SchedulerHook.REQUEST_STATS,
                )
            stats: list[RequestStats] = []
            for item in C_SchedulerHook.REQUEST_STATS.values():
                if item.rid is not None and item.input_length > 0:
                    stats.append(item)

            stats = sorted(stats, key=lambda req: req.created_time)

            output_dir = Envs.output_dir()
            os.makedirs(output_dir, exist_ok=True)

            ProfileReqOutput = getattr(
                importlib.import_module("sglang.srt.managers.io_struct"),
                "ProfileReqOutput",
            )

            if len(stats) == 0:
                logger.info(
                    "Profile requested with no completed simulation stats; keeping PD backend/state intact."
                )
                return ProfileReqOutput(
                    True,
                    json.dumps(
                        {
                            "total_request": 0,
                            "output_directory": output_dir,
                        }
                    ),
                )

            if len(stats) > 0:
                # Remove warmup requests.
                if len(stats) > Envs.num_warmup():
                    metrics_stats = stats[Envs.num_warmup() :]
                else:
                    metrics_stats = stats

                min_created_time = metrics_stats[0].created_time
                # Align timestamps
                from hisim.simulation.pd_metrics import shift_pd_time_origin

                for item in stats:
                    item.created_time -= min_created_time
                    item.queue_start -= min_created_time
                    item.queue_end -= min_created_time
                    item.last_event_time -= min_created_time
                    shift_pd_time_origin(item, min_created_time)

                metrics = calc_metrics(metrics_stats)
                metrics["time_cost"] = time.time() - C_SchedulerHook.LAST_FLUSH_TS

                try:
                    with open(f"{output_dir}/metrics.json", "w") as f:
                        f.write(json.dumps(metrics, cls=CustomJsonEncoder) + "\n")

                    with open(f"{output_dir}/iteration.jsonl", "w") as f:
                        for item in C_SchedulerHook.ITERATION_STATS:
                            f.write(json.dumps(item) + "\n")

                    with open(f"{output_dir}/request.jsonl", "w") as f:
                        for item in stats:
                            f.write(json.dumps(asdict(item)) + "\n")

                    logger.info(f"Simulation results saved to {output_dir}.")

                except Exception as e:
                    logger.error(f"Failed to dump results. Error: {e}")
            else:
                logger.warning("No request statistics available.")

            StateManager.reset()
            C_SchedulerHook.REQUEST_STATS.clear()
            C_SchedulerHook.ITERATION_STATS.clear()
            C_SchedulerHook.PD_REQUEST_STATES.clear()
            C_SchedulerHook.PD_PREFILL_KV_SERVICE.clear()
            C_SchedulerHook.PD_CHUNK_ACCUM.clear()
            C_SchedulerHook.LAST_CPU_TS = 0
            C_SchedulerHook.LAST_FLUSH_TS = time.time()
            C_SchedulerHook.OFFLINE_RECV_ALL_REQUEST = False

            # Phase 5c.2: tear down PD backend workers (BackendB only).
            # atexit also fires shutdown_pd_backend; idempotent.
            if C_SchedulerHook.PD_BACKEND is not None:
                from hisim.simulation.pd_runtime import shutdown_pd_backend

                shutdown_pd_backend(C_SchedulerHook.PD_BACKEND)
                C_SchedulerHook.PD_BACKEND = None

            result = {
                "total_request": len(stats),
                "output_directory": output_dir,
            }

            return ProfileReqOutput(True, json.dumps(result))

        target.event_loop_overlap = override_event_loop_overlap
        target.__init__ = wrapped_init
        target.recv_requests = wrapped_recv_requests
        target.get_new_batch_prefill = wrapped_get_new_batch_prefill
        target.update_running_batch = wrapped_update_running_batch
        target.run_batch = wrapped_run_batch
        target.process_batch_result = wrapped_process_batch_result
        target.profile = wrapped_profile
