# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from collections import OrderedDict
from collections.abc import Collection, Iterable, Sequence
from typing import Literal

from typing_extensions import override

from vllm.distributed.kv_events import MEDIUM_CPU
from vllm.distributed.kv_transfer.kv_connector.v1.offloading.metrics import (
    OffloadingConnectorStats,
)
from vllm.v1.kv_offload.base import (
    LoadStoreSpec,
    LookupResult,
    OffloadingEvent,
    OffloadingManager,
    OffloadKey,
    PrepareStoreOutput,
    ReqContext,
    RequestOffloadingContext,
    get_offload_group_idx,
)
from vllm.v1.kv_offload.cpu.common import (
    CPULoadStoreSpec,
    CPUOffloadingMetrics,
)
from vllm.v1.kv_offload.cpu.policies.arc import ARCCachePolicy
from vllm.v1.kv_offload.cpu.policies.base import BlockStatus, CachePolicy
from vllm.v1.kv_offload.cpu.policies.lru import LRUCachePolicy

_CACHE_POLICIES: dict[str, type[CachePolicy]] = {
    "lru": LRUCachePolicy,
    "arc": ARCCachePolicy,
}


class CPUOffloadingManager(OffloadingManager):
    """
    An OffloadingManager with a pluggable CachePolicy (LRU or ARC).

    The manager owns all shared logic: ref-counting, event emission,
    block pool management, and the prepare_store/complete_store skeletons.
    Policy-specific block organization and eviction decisions are delegated
    to the CachePolicy implementation.
    """

    def __init__(
        self,
        num_blocks: int,
        cache_policy: Literal["lru", "arc"] = "lru",
        enable_events: bool = False,
        store_threshold: int = 1,
        max_tracker_size: int = 64_000,
    ):
        self.medium: str = MEDIUM_CPU
        self._num_blocks: int = num_blocks
        self._num_allocated_blocks: int = 0
        self._free_list: list[int] = []
        self.events: list[OffloadingEvent] | None = [] if enable_events else None
        policy_cls = _CACHE_POLICIES.get(cache_policy)
        if policy_cls is None:
            raise ValueError(
                f"Unknown cache policy: {cache_policy!r}. "
                f"Supported: {list(_CACHE_POLICIES)}"
            )
        self._policy: CachePolicy = policy_cls(cache_capacity=num_blocks)
        # Track the number of blocks in the cache that are evictable. i.e. ref_cnt 0.
        self._num_evictable_cache_blocks: int = 0
        # Track blocks with an in-flight store (ref_cnt -1, not yet completed).
        self._num_write_pending_blocks: int = 0

        self.store_threshold: int = store_threshold
        self.max_tracker_size: int = max_tracker_size
        self.stores_skipped_in_current_batch: int = 0
        self.allocation_sizes_in_current_batch: list[int] = []

        # Number of block references. It is ordered so can evict the LRU entry in O(1).
        self.counts: OrderedDict[OffloadKey, int] | None = (
            OrderedDict() if store_threshold >= 2 else None
        )

    # --- block pool ---

    def _get_num_free_blocks(self) -> int:
        return len(self._free_list) + self._num_blocks - self._num_allocated_blocks

    def _allocate_blocks(self, keys: list[OffloadKey]) -> list[BlockStatus]:
        num_fresh = min(len(keys), self._num_blocks - self._num_allocated_blocks)
        num_reused = len(keys) - num_fresh
        assert len(self._free_list) >= num_reused

        # allocate fresh blocks
        blocks: list[BlockStatus] = []
        for _ in range(num_fresh):
            blocks.append(BlockStatus(self._num_allocated_blocks))
            self._num_allocated_blocks += 1

        # allocate reused blocks
        for _ in range(num_reused):
            blocks.append(BlockStatus(self._free_list.pop()))
        return blocks

    def _free_block(self, block: BlockStatus) -> None:
        self._free_list.append(block.block_id)

    def _get_load_store_spec(
        self,
        keys: Iterable[OffloadKey],
        blocks: Iterable[BlockStatus],
    ) -> CPULoadStoreSpec:
        return CPULoadStoreSpec([block.block_id for block in blocks])

    # --- OffloadingManager interface ---

    @override
    def on_new_request(self, req_context: ReqContext) -> RequestOffloadingContext:
        return RequestOffloadingContext()

    @override
    def lookup(self, key: OffloadKey, req_context: ReqContext) -> LookupResult:
        if self.counts is not None:
            if key in self.counts:
                self.counts.move_to_end(key)
                self.counts[key] += 1
            else:
                if len(self.counts) >= self.max_tracker_size:
                    self.counts.popitem(last=False)
                self.counts[key] = 1
        block = self._policy.get(key)
        if block is None:
            return LookupResult.MISS
        if not block.is_ready:
            return LookupResult.HIT_PENDING
        return LookupResult.HIT

    @override
    def prepare_load(
        self,
        keys: Collection[OffloadKey],
        req_context: ReqContext,
    ) -> LoadStoreSpec:
        blocks = []
        for key in keys:
            block = self._policy.get(key)
            assert block is not None, f"Block {key!r} not found in cache"
            assert block.is_ready, f"Block {key!r} is not ready for reading"
            if block.ref_cnt == 0:
                self._policy.mark_non_evictable(key)
                self._num_evictable_cache_blocks -= 1  # ref_cnt 0 -> 1
                assert self._num_evictable_cache_blocks >= 0
            block.ref_cnt += 1
            blocks.append(block)
        return self._get_load_store_spec(keys, blocks)

    @override
    def touch(self, keys: Collection[OffloadKey], req_context: ReqContext) -> None:
        self._policy.touch(keys, req_context)

    @override
    def complete_load(
        self, keys: Collection[OffloadKey], req_context: ReqContext
    ) -> None:
        for key in keys:
            block = self._policy.get(key)
            assert block is not None, f"Block {key!r} not found"
            assert block.ref_cnt > 0, f"Block {key!r} ref_cnt is already 0"
            block.ref_cnt -= 1
            if block.ref_cnt == 0:
                self._num_evictable_cache_blocks += 1  # ref_cnt 1 -> 0
                self._policy.mark_evictable(key)

    @override
    def prepare_store(
        self,
        keys: Collection[OffloadKey],
        req_context: ReqContext,
    ) -> PrepareStoreOutput | None:
        if self.counts is not None:
            num_keys = len(keys)
            keys = [k for k in keys if self.counts.get(k, 0) >= self.store_threshold]
            self.stores_skipped_in_current_batch += num_keys - len(keys)
        # filter out blocks that are already stored
        keys_to_store = [k for k in keys if self._policy.get(k) is None]

        if not keys_to_store:
            return PrepareStoreOutput(
                keys_to_store=[],
                store_spec=self._get_load_store_spec([], []),
                evicted_keys=[],
            )

        self.allocation_sizes_in_current_batch.append(len(keys_to_store))
        num_blocks_to_evict = len(keys_to_store) - self._get_num_free_blocks()

        to_evict: list[OffloadKey] = []
        if num_blocks_to_evict > 0:
            if num_blocks_to_evict > self._num_evictable_cache_blocks:
                # Eviction will fail.
                return None
            # There is a still a chance for eviction failure as some of the
            # idle blocks might be in the protected list.

            # Blocks from the original input are excluded from eviction candidates:
            # a block that was already stored must remain in the cache after this call.
            protected = set(keys)
            evicted = self._policy.evict(num_blocks_to_evict, protected)
            if evicted is None:
                return None

            # cache-policy removes only idle blocks.
            self._num_evictable_cache_blocks -= len(evicted)
            assert self._num_evictable_cache_blocks >= 0

            for key, block in evicted:
                self._free_block(block)
                to_evict.append(key)

        if to_evict and self.events is not None:
            self.events.append(
                OffloadingEvent(
                    keys=to_evict,
                    medium=self.medium,
                    removed=True,
                )
            )

        blocks = self._allocate_blocks(keys_to_store)
        assert len(blocks) == len(keys_to_store), (
            "Block pool did not allocate the expected number of blocks"
        )

        for key, block in zip(keys_to_store, blocks):
            self._policy.insert(key, block)
        self._num_write_pending_blocks += len(keys_to_store)

        # build store specs for allocated blocks
        store_spec = self._get_load_store_spec(keys_to_store, blocks)

        return PrepareStoreOutput(
            keys_to_store=keys_to_store,
            store_spec=store_spec,
            evicted_keys=to_evict,
        )

    @override
    def complete_store(
        self,
        keys: Collection[OffloadKey],
        req_context: ReqContext,
        success: bool = True,
    ) -> None:
        stored_keys: list[OffloadKey] = []

        if success:
            for key in keys:
                block = self._policy.get(key)
                if block is not None and not block.is_ready:
                    block.ref_cnt = 0
                    self._num_write_pending_blocks -= 1
                    self._num_evictable_cache_blocks += 1
                    self._policy.mark_evictable(key)
                    stored_keys.append(key)
        else:
            for key in keys:
                block = self._policy.get(key)
                if block is not None and not block.is_ready:
                    self._num_write_pending_blocks -= 1
                    self._policy.remove(key)
                    self._free_block(block)

        if stored_keys and self.events is not None:
            self.events.append(
                OffloadingEvent(
                    keys=stored_keys,
                    medium=self.medium,
                    removed=False,
                )
            )

    @override
    def reset_cache(self) -> None:
        # Clear ALL blocks unconditionally. The scheduler's _stale_job_threshold
        # guarantees that complete_load / complete_store are never called for
        # pre-reset jobs, so no lazy cleanup is needed. The scheduler also
        # flushes in-flight load job IDs to the workers before any new stores
        # can begin, preventing a cross-direction data race on reused offload block IDs.
        self._policy.clear()
        self._num_evictable_cache_blocks = 0
        self._num_write_pending_blocks = 0

        self._free_list.clear()
        self._num_allocated_blocks = 0

    @override
    def take_events(self) -> Iterable[OffloadingEvent]:
        if self.events is not None:
            yield from self.events
            self.events.clear()

    def get_stats(self) -> OffloadingConnectorStats | None:
        stats = OffloadingConnectorStats()

        # Compute cache usage.
        num_used = (
            self._num_allocated_blocks
            - len(self._free_list)
            - self._num_evictable_cache_blocks
        )
        usage = num_used / self._num_blocks if self._num_blocks > 0 else 0.0
        stats.set_gauge(CPUOffloadingMetrics.CPU_CACHE_USAGE_PERC, usage)

        for allocation_size in self.allocation_sizes_in_current_batch:
            stats.observe_histogram(
                CPUOffloadingMetrics.CPU_ALLOCATION_SIZE, allocation_size
            )
        self.allocation_sizes_in_current_batch.clear()

        write_usage = (
            self._num_write_pending_blocks / self._num_blocks
            if self._num_blocks > 0
            else 0.0
        )
        read_usage = max(usage - write_usage, 0.0)
        stats.set_gauge(CPUOffloadingMetrics.CPU_CACHE_WRITE_USAGE_PERC, write_usage)
        stats.set_gauge(CPUOffloadingMetrics.CPU_CACHE_READ_USAGE_PERC, read_usage)

        if self.store_threshold >= 2:
            stats.increase_counter(
                CPUOffloadingMetrics.STORES_SKIPPED,
                self.stores_skipped_in_current_batch,
            )
            self.stores_skipped_in_current_batch = 0

        return stats


class GroupRoutedCPUOffloadingManager(OffloadingManager):
    """Routes offloading operations to per-KV-group CPUOffloadingManagers.

    Used when KV cache groups have heterogeneous per-block byte costs
    (e.g. hierarchical mamba state blocks vs attention blocks). Each group
    gets its own CPU block pool and eviction policy, so churn in one group
    cannot evict the other's entries, and block ids are group-local,
    matching the per-group-sized CPU tensors on the worker side.

    Multi-key operations partition their keys by the group index embedded
    in each OffloadKey. Key order within each group is preserved, and the
    connector emits keys grouped in group-index order, so concatenating
    per-group spec block ids reproduces the input key order.
    """

    def __init__(self, managers: "Sequence[CPUOffloadingManager]"):
        assert len(managers) > 0
        self.medium: str = MEDIUM_CPU
        # Indexed by group; same-cost-class groups share a manager instance.
        self._managers = tuple(managers)
        # Distinct instances, for aggregate operations (stats/reset/events).
        self._unique_managers = tuple(dict.fromkeys(managers))

    def _manager_of(self, key: OffloadKey) -> CPUOffloadingManager:
        return self._managers[get_offload_group_idx(key)]

    @staticmethod
    def _group_runs(
        keys: Iterable[OffloadKey],
    ) -> list[tuple[int, list[OffloadKey]]]:
        """Split keys into contiguous same-group runs, preserving order."""
        runs: list[tuple[int, list[OffloadKey]]] = []
        for key in keys:
            group_idx = get_offload_group_idx(key)
            if runs and runs[-1][0] == group_idx:
                runs[-1][1].append(key)
            else:
                runs.append((group_idx, [key]))
        return runs

    @override
    def on_new_request(self, req_context: ReqContext) -> RequestOffloadingContext:
        return RequestOffloadingContext()

    @override
    def lookup(self, key: OffloadKey, req_context: ReqContext) -> LookupResult:
        return self._manager_of(key).lookup(key, req_context)

    @override
    def prepare_load(
        self,
        keys: Collection[OffloadKey],
        req_context: ReqContext,
    ) -> CPULoadStoreSpec:
        block_ids: list[int] = []
        for group_idx, run in self._group_runs(keys):
            spec = self._managers[group_idx].prepare_load(run, req_context)
            block_ids.extend(spec.block_ids.tolist())
        return CPULoadStoreSpec(block_ids)

    @override
    def touch(self, keys: Collection[OffloadKey], req_context: ReqContext) -> None:
        for group_idx, run in self._group_runs(keys):
            self._managers[group_idx].touch(run, req_context)

    @override
    def complete_load(
        self, keys: Collection[OffloadKey], req_context: ReqContext
    ) -> None:
        for group_idx, run in self._group_runs(keys):
            self._managers[group_idx].complete_load(run, req_context)

    @override
    def prepare_store(
        self,
        keys: Collection[OffloadKey],
        req_context: ReqContext,
    ) -> PrepareStoreOutput | None:
        prepared: list[tuple[int, PrepareStoreOutput]] = []
        for group_idx, run in self._group_runs(keys):
            output = self._managers[group_idx].prepare_store(run, req_context)
            if output is None:
                # Roll back groups already prepared in this call so no block
                # is left pending a store that will never be submitted.
                # Entries a successful earlier prepare_store already EVICTED
                # cannot be restored — that is bounded cache loss (the store
                # never happens), not corruption.
                for done_idx, done_output in prepared:
                    if done_output.keys_to_store:
                        self._managers[done_idx].complete_store(
                            done_output.keys_to_store, req_context, success=False
                        )
                return None
            prepared.append((group_idx, output))

        keys_to_store: list[OffloadKey] = []
        block_ids: list[int] = []
        evicted_keys: list[OffloadKey] = []
        for _, output in prepared:
            keys_to_store.extend(output.keys_to_store)
            block_ids.extend(output.store_spec.block_ids.tolist())
            evicted_keys.extend(output.evicted_keys)
        return PrepareStoreOutput(
            keys_to_store=keys_to_store,
            store_spec=CPULoadStoreSpec(block_ids),
            evicted_keys=evicted_keys,
        )

    @override
    def complete_store(
        self,
        keys: Collection[OffloadKey],
        req_context: ReqContext,
        success: bool = True,
    ) -> None:
        for group_idx, run in self._group_runs(keys):
            self._managers[group_idx].complete_store(run, req_context, success)

    @override
    def reset_cache(self) -> None:
        for manager in self._unique_managers:
            manager.reset_cache()

    @override
    def take_events(self) -> Iterable[OffloadingEvent]:
        for manager in self._unique_managers:
            yield from manager.take_events()

    @override
    def get_stats(self) -> OffloadingConnectorStats | None:
        # Capacity-weighted aggregation over the sub-pools, mirroring the
        # single-pool math in CPUOffloadingManager.get_stats.
        stats = OffloadingConnectorStats()
        total_blocks = sum(m._num_blocks for m in self._unique_managers)
        num_used = sum(
            m._num_allocated_blocks - len(m._free_list) - m._num_evictable_cache_blocks
            for m in self._unique_managers
        )
        usage = num_used / total_blocks if total_blocks > 0 else 0.0
        stats.set_gauge(CPUOffloadingMetrics.CPU_CACHE_USAGE_PERC, usage)

        for manager in self._unique_managers:
            for allocation_size in manager.allocation_sizes_in_current_batch:
                stats.observe_histogram(
                    CPUOffloadingMetrics.CPU_ALLOCATION_SIZE, allocation_size
                )
            manager.allocation_sizes_in_current_batch.clear()

        write_pending = sum(m._num_write_pending_blocks for m in self._unique_managers)
        write_usage = write_pending / total_blocks if total_blocks > 0 else 0.0
        read_usage = max(usage - write_usage, 0.0)
        stats.set_gauge(CPUOffloadingMetrics.CPU_CACHE_WRITE_USAGE_PERC, write_usage)
        stats.set_gauge(CPUOffloadingMetrics.CPU_CACHE_READ_USAGE_PERC, read_usage)

        # Match the single-pool emission cadence: increment (0 included)
        # whenever the metric is registered, i.e. a store threshold >= 2.
        if any(m.store_threshold >= 2 for m in self._unique_managers):
            stores_skipped = 0
            for manager in self._unique_managers:
                if manager.store_threshold >= 2:
                    stores_skipped += manager.stores_skipped_in_current_batch
                    manager.stores_skipped_in_current_batch = 0
            stats.increase_counter(CPUOffloadingMetrics.STORES_SKIPPED, stores_skipped)

        return stats
