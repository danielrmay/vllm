# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from typing import Any

from typing_extensions import override

from vllm.platforms import current_platform
from vllm.utils.math_utils import round_up
from vllm.v1.kv_offload.base import (
    CanonicalKVCaches,
    OffloadingCounterMetadata,
    OffloadingGaugeMetadata,
    OffloadingHistogramMetadata,
    OffloadingManager,
    OffloadingMetricMetadata,
    OffloadingSpec,
    OffloadingWorker,
)
from vllm.v1.kv_offload.config import OffloadingConfig
from vllm.v1.kv_offload.cpu.common import CPUOffloadingMetrics
from vllm.v1.kv_offload.cpu.gpu_worker import CPUOffloadingWorker
from vllm.v1.kv_offload.cpu.manager import (
    CPUOffloadingManager,
    GroupRoutedCPUOffloadingManager,
)


def _fmt_bytes(num_bytes: float) -> str:
    """Adaptive units: '0.00 GiB (< one 0.00 GiB chunk)' is useless."""
    for unit, scale in (("GiB", 2**30), ("MiB", 2**20), ("KiB", 2**10)):
        if num_bytes >= scale:
            return f"{num_bytes / scale:.2f} {unit}"
    return f"{num_bytes:.0f} B"


class CPUOffloadingSpec(OffloadingSpec):
    BLOCK_SIZE_ALIGNMENT = 1

    @classmethod
    def build_metric_definitions(
        cls, extra_config: dict[str, Any]
    ) -> dict[str, OffloadingMetricMetadata]:
        definitions: dict[str, OffloadingMetricMetadata] = {
            CPUOffloadingMetrics.CPU_CACHE_USAGE_PERC: OffloadingGaugeMetadata(
                documentation=(
                    "Fraction of CPU KV-cache space currently pinned by active "
                    "transfers (0.0 = idle, 1.0 = saturated). Sustained high "
                    "values indicate transfers (stores or promotions) may be "
                    "dropped due to insufficient capacity."
                ),
            ),
            CPUOffloadingMetrics.CPU_CACHE_WRITE_USAGE_PERC: OffloadingGaugeMetadata(
                documentation=(
                    "Fraction of CPU KV-cache space currently pinned by "
                    "in-flight stores that have not yet "
                    "completed (0.0 = idle, 1.0 = saturated)."
                ),
            ),
            CPUOffloadingMetrics.CPU_CACHE_READ_USAGE_PERC: OffloadingGaugeMetadata(
                documentation=(
                    "Fraction of CPU KV-cache space currently pinned by "
                    "in-flight loads that have not yet "
                    "completed (0.0 = idle, 1.0 = saturated)."
                ),
            ),
            CPUOffloadingMetrics.CPU_ALLOCATION_SIZE: OffloadingHistogramMetadata(
                documentation=(
                    "Histogram of the number of CPU blocks requested by each "
                    "KV offload prepare_store call."
                ),
                buckets=(1, 4, 16, 64, 256, 1024, 4096, 16384, 65536, 262144),
            ),
        }
        store_threshold = int(extra_config.get("store_threshold", 0))
        if store_threshold >= 2:
            definitions[CPUOffloadingMetrics.STORES_SKIPPED] = (
                OffloadingCounterMetadata(
                    documentation=(
                        "Number of KV offload stores skipped because the reuse "
                        "threshold was not reached."
                    ),
                )
            )
        return definitions

    def __init__(self, config: OffloadingConfig):
        super().__init__(config)

        cpu_bytes_to_use = self.extra_config.get("cpu_bytes_to_use")
        if not cpu_bytes_to_use:
            raise Exception(
                "cpu_bytes_to_use must be specified in kv_connector_extra_config"
            )

        world_size = config.parallel.world_size
        self.num_blocks = 0
        self.kv_bytes_per_chunk = 0
        self.cpu_page_size_per_worker = 0

        # Per-group sizing: when KV cache groups have heterogeneous per-block
        # byte costs (e.g. hierarchical mamba state blocks vs attention
        # blocks), a single uniform pool mis-sizes catastrophically (the CPU
        # region is allocated per canonical tensor). Give each group its own
        # pool, sized by its share of the GPU byte rate.
        group_kv_bytes = [g.kv_bytes_per_block for g in config.groups]
        self.num_blocks_per_group: list[int] | None = None
        self._class_of_group: list[int] | None = None
        if len(set(group_kv_bytes)) > 1 and all(b > 0 for b in group_kv_bytes):
            # Pools are per COST CLASS, not per group: interleaved hybrids
            # split same-spec layers into multiple KV groups that share
            # canonical tensors, so same-cost groups must share one pool
            # (one id space, one eviction policy, one tensor row count).
            classes = sorted(set(group_kv_bytes))
            self._class_of_group = [classes.index(b) for b in group_kv_bytes]
            class_rates = [0.0] * len(classes)
            for group in config.groups:
                class_rates[classes.index(group.kv_bytes_per_block)] += (
                    group.kv_bytes_per_block / group.tokens_per_block
                )
            total_rate = sum(class_rates)
            class_chunk_bytes = [
                round_up(
                    class_bytes * world_size * self.blocks_per_chunk,
                    self.BLOCK_SIZE_ALIGNMENT,
                )
                for class_bytes in classes
            ]
            # The budget that gives EVERY class at least one chunk (the
            # binding class maximizes chunk_bytes / rate); suggesting only
            # the first failing class's requirement would leave the user
            # bumping the budget repeatedly.
            sufficient_budget = max(
                total_rate * chunk_bytes / rate
                for chunk_bytes, rate in zip(class_chunk_bytes, class_rates)
            )
            num_blocks_per_class = []
            for class_bytes, chunk_bytes, rate in zip(
                classes, class_chunk_bytes, class_rates
            ):
                class_budget = int(cpu_bytes_to_use) * rate / total_rate
                num_class_blocks = int(class_budget // chunk_bytes)
                if num_class_blocks < 1:
                    # A zero-capacity class would make its prepare_store
                    # return None, and the router's all-or-nothing contract
                    # then rejects EVERY store — silently disabling the whole
                    # CPU tier. Fail loudly with the sizing math instead.
                    raise ValueError(
                        "CPU offloading budget too small: cost class with "
                        f"{class_bytes} bytes/block gets "
                        f"{_fmt_bytes(class_budget)} "
                        f"(< one {_fmt_bytes(chunk_bytes)} chunk). "
                        "Increase kv_offloading_size to at least "
                        f"{_fmt_bytes(sufficient_budget)} "
                        f"({int(sufficient_budget) + 1} bytes; sizes "
                        "every cost class for at least one chunk)."
                    )
                num_blocks_per_class.append(num_class_blocks)
            self.num_blocks_per_group = [
                num_blocks_per_class[c] for c in self._class_of_group
            ]
            self.num_blocks = sum(num_blocks_per_class)
        elif config.worker_kv_bytes_per_block > 0 and world_size > 0:
            kv_bytes_per_block = config.worker_kv_bytes_per_block * world_size
            kv_bytes_per_chunk = kv_bytes_per_block * self.blocks_per_chunk

            # calculate cpu_page_size_per_worker
            self.cpu_page_size_per_worker = kv_bytes_per_chunk // world_size

            # calculate num_blocks
            aligned_kv_bytes_per_chunk = round_up(
                kv_bytes_per_chunk, self.BLOCK_SIZE_ALIGNMENT
            )
            self.num_blocks = int(cpu_bytes_to_use) // aligned_kv_bytes_per_chunk

            # Expose aligned_kv_bytes_per_chunk as
            # kv_bytes_per_chunk. Note that this might contain
            # some padding. i.e. each offloaded block is of the form,
            # |--- W0-B0---|---- W1-B0---| ... |---- Wn-B0---| *** maybe-pad *** |
            self.kv_bytes_per_chunk = aligned_kv_bytes_per_chunk

        # scheduler-side
        self._manager: OffloadingManager | None = None

        # worker-side
        self._worker: CPUOffloadingWorker | None = None

        self.eviction_policy: str = self.extra_config.get("eviction_policy", "lru")

    @override
    def get_manager(self) -> OffloadingManager:
        if not self._manager:
            # store_threshold: how many times a block must appear in lookup()
            # before it is eligible for CPU offloading.  Values < 2 disable
            # filtering (a threshold of 1 equals no filter; 0 is the default).
            store_threshold = int(self.extra_config.get("store_threshold", 0))

            # Maximum entries in the internal tracker's LRU table.
            max_tracker_size = int(self.extra_config.get("max_tracker_size", 64_000))

            def make_manager(num_blocks: int) -> CPUOffloadingManager:
                return CPUOffloadingManager(
                    num_blocks=num_blocks,
                    cache_policy=self.eviction_policy,  # type: ignore[arg-type]
                    enable_events=self.kv_events_config.enable_kv_cache_events,
                    store_threshold=store_threshold,
                    max_tracker_size=max_tracker_size,
                )

            if self.num_blocks_per_group is not None:
                # Heterogeneous block costs: one pool per COST CLASS, with
                # same-class groups sharing the manager instance (shared id
                # space matching the shared canonical tensors).
                assert self._class_of_group is not None
                class_managers: dict[int, CPUOffloadingManager] = {}
                per_group_managers = []
                for group_idx, class_idx in enumerate(self._class_of_group):
                    if class_idx not in class_managers:
                        class_managers[class_idx] = make_manager(
                            self.num_blocks_per_group[group_idx]
                        )
                    per_group_managers.append(class_managers[class_idx])
                self._manager = GroupRoutedCPUOffloadingManager(per_group_managers)
            else:
                self._manager = make_manager(self.num_blocks)
        return self._manager

    def create_worker(self, kv_caches: CanonicalKVCaches) -> CPUOffloadingWorker:
        return CPUOffloadingWorker(
            kv_caches=kv_caches,
            blocks_per_chunk=self.blocks_per_chunk,
            num_cpu_blocks=(
                self.num_blocks_per_group
                if self.num_blocks_per_group is not None
                else self.num_blocks
            ),
            max_pin_fraction=self._max_pin_fraction(),
        )

    def _max_pin_fraction(self) -> float:
        """Fraction of AVAILABLE host memory (at init, so restart order on a
        busy host matters) the pinned CPU region may claim; values above 1.0
        effectively disable the guard for deliberate large-pin deployments.
        Set via kv_connector_extra_config["max_pin_fraction"]; default 0.5."""
        fraction = float(self.extra_config.get("max_pin_fraction", 0.5))
        if fraction <= 0:
            raise ValueError(f"max_pin_fraction must be positive, got {fraction}")
        return fraction

    @override
    def get_worker(self, kv_caches: CanonicalKVCaches) -> OffloadingWorker:
        if not self._worker:
            if not (current_platform.is_cuda_alike() or current_platform.is_xpu()):
                raise Exception(
                    "CPU Offloading is currently only supported on CUDA-alike "
                    "and XPU GPUs"
                )
            self._worker = self.create_worker(kv_caches)

        assert self._worker is not None
        return self._worker
