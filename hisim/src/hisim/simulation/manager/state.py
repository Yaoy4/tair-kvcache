class StateManager:
    _iteration: int = 0
    _global_clock: float = 0
    _last_inference_dur: float = 0
    _current_inference_dur: float = 0
    _hicache_l2_load_dur: float = 0
    _hicache_l2_backup_dur: float = 0

    # PD disaggregation: the prefill (P) and decode (D) phases run on separate
    # pools of instances, each with its own virtual clock. KV cache readiness is
    # tracked per migration (returned by complete_kv_migration and passed into
    # start_decode) so that a decode only waits on its own P->D transfer rather
    # than on unrelated, later migrations.
    _num_prefill_instances: int = 1
    _num_decode_instances: int = 1
    _prefill_clocks: list = [0.0]
    _decode_clocks: list = [0.0]
    # Aggregate latest migration completion time, kept for metrics/debugging only.
    _last_kv_ready_time: float = 0.0

    @classmethod
    def reset(cls):
        cls._iteration = 0
        cls._global_clock = 0
        cls._last_inference_dur = 0
        cls._current_inference_dur = 0
        cls._hicache_l2_backup_dur = 0
        cls._hicache_l2_load_dur = 0
        cls._num_prefill_instances = 1
        cls._num_decode_instances = 1
        cls._prefill_clocks = [0.0]
        cls._decode_clocks = [0.0]
        cls._last_kv_ready_time = 0.0

    @classmethod
    def init_pd_timelines(cls, num_prefill: int = 1, num_decode: int = 1) -> None:
        if num_prefill < 1 or num_decode < 1:
            raise ValueError(
                "PD instance counts must be >= 1, got "
                f"num_prefill={num_prefill}, num_decode={num_decode}"
            )
        cls._num_prefill_instances = num_prefill
        cls._num_decode_instances = num_decode
        cls._prefill_clocks = [0.0] * num_prefill
        cls._decode_clocks = [0.0] * num_decode
        cls._last_kv_ready_time = 0.0

    @classmethod
    def get_prefill_clock(cls, idx: int) -> float:
        return cls._prefill_clocks[idx]

    @classmethod
    def get_decode_clock(cls, idx: int) -> float:
        return cls._decode_clocks[idx]

    @classmethod
    def get_last_kv_ready_time(cls) -> float:
        return cls._last_kv_ready_time

    @classmethod
    def pick_prefill_instance(cls) -> int:
        """Return the index of the least-loaded (earliest free) P instance."""
        return min(
            range(len(cls._prefill_clocks)), key=lambda i: cls._prefill_clocks[i]
        )

    @classmethod
    def pick_decode_instance(cls) -> int:
        """Return the index of the least-loaded (earliest free) D instance."""
        return min(
            range(len(cls._decode_clocks)), key=lambda i: cls._decode_clocks[i]
        )

    @classmethod
    def step_prefill_clock(cls, idx: int, dur: float) -> float:
        """Advance a P instance clock by its prefill compute duration."""
        cls._prefill_clocks[idx] += dur
        return cls._prefill_clocks[idx]

    @classmethod
    def complete_kv_migration(cls, prefill_idx: int, transfer_dur: float) -> float:
        """Compute when the KV cache produced by ``prefill_idx`` becomes available
        on the decode side.

        The transfer runs over the network after prefill compute completes and is
        modeled as a separate resource, so it does not occupy (advance) the P
        instance clock. Returns the absolute KV-ready timestamp, which must be
        passed to :meth:`start_decode`.
        """
        ready_time = cls._prefill_clocks[prefill_idx] + transfer_dur
        cls._last_kv_ready_time = max(cls._last_kv_ready_time, ready_time)
        return ready_time

    @classmethod
    def start_decode(
        cls, decode_idx: int, dur: float, kv_ready_time: float = 0.0
    ) -> tuple:
        """Schedule a decode step on a D instance.

        A decode cannot begin before its KV cache migration from the P instance
        completes, so it starts at ``max(decode_clock, kv_ready_time)``. Returns
        the ``(start, end)`` timestamps of the decode step.
        """
        start = max(cls._decode_clocks[decode_idx], kv_ready_time)
        end = start + dur
        cls._decode_clocks[decode_idx] = end
        return start, end

    @classmethod
    def inc_iteration(cls) -> None:
        cls._iteration += 1

    @classmethod
    def get_iteration(cls) -> int:
        return cls._iteration

    @classmethod
    def inc_hicache_l2_load_dur(cls, dur: float) -> None:
        cls._hicache_l2_load_dur += dur

    @classmethod
    def inc_hicache_l2_backup_dur(cls, dur: float) -> None:
        cls._hicache_l2_backup_dur += dur

    @classmethod
    def pop_hicache_l2_load_dur(cls) -> float:
        dur = cls._hicache_l2_load_dur
        cls._hicache_l2_load_dur = 0
        return dur

    @classmethod
    def pop_hicache_l2_backup_dur(cls) -> float:
        dur = cls._hicache_l2_backup_dur
        cls._hicache_l2_backup_dur = 0
        return dur

    @classmethod
    def get_global_clock(cls) -> float:
        return cls._global_clock

    @classmethod
    def step_global_clock(cls, dur: float) -> None:
        cls._global_clock += dur

    @classmethod
    def set_global_clock(cls, clock: float) -> None:
        cls._global_clock = clock

    @classmethod
    def set_current_inference_dur(cls, dur: float) -> None:
        cls._last_inference_dur = cls._current_inference_dur
        cls._current_inference_dur = dur

    @classmethod
    def get_last_inference_dur(cls) -> float:
        return cls._last_inference_dur

    @classmethod
    def get_current_inference_dur(cls) -> float:
        return cls._current_inference_dur
