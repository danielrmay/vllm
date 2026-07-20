# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Direct unit coverage for GroupRoutedCPUOffloadingManager: run
partitioning, the all-or-nothing prepare_store rollback, and stats
aggregation. These paths otherwise only run inside the GPU e2e tests."""

from vllm.v1.kv_offload.base import LookupResult, ReqContext, make_offload_key
from vllm.v1.kv_offload.cpu.common import CPUOffloadingMetrics
from vllm.v1.kv_offload.cpu.manager import (
    CPUOffloadingManager,
    GroupRoutedCPUOffloadingManager,
)

_CTX = ReqContext(req_id="")


def key(group_idx: int, block_hash: int) -> bytes:
    return make_offload_key(str(block_hash).encode(), group_idx)


def make_router(
    blocks_per_group: list[int], store_threshold: int = 0
) -> GroupRoutedCPUOffloadingManager:
    managers = [
        CPUOffloadingManager(
            num_blocks=n,
            cache_policy="lru",
            enable_events=False,
            store_threshold=store_threshold,
            max_tracker_size=64_000,
        )
        for n in blocks_per_group
    ]
    return GroupRoutedCPUOffloadingManager(managers)


def test_group_runs_partition_preserves_order():
    """Keys split into contiguous same-group runs; key order within each
    run must be the input order (the connector emits keys grouped in
    group-index order, and per-group block ids are concatenated back)."""
    keys = [key(0, 1), key(0, 2), key(1, 3), key(1, 4), key(0, 5)]
    runs = GroupRoutedCPUOffloadingManager._group_runs(keys)
    assert [(g, len(run)) for g, run in runs] == [(0, 2), (1, 2), (0, 1)]
    assert runs[0][1] == keys[0:2]
    assert runs[1][1] == keys[2:4]
    assert runs[2][1] == keys[4:5]


def test_prepare_store_routes_and_concatenates_block_ids():
    router = make_router([4, 4])
    keys = [key(0, 1), key(1, 2), key(1, 3)]
    output = router.prepare_store(keys, _CTX)
    assert output is not None
    assert output.keys_to_store == keys
    # 1 block from group 0's pool + 2 from group 1's, in input key order.
    assert len(output.store_spec.block_ids) == 3


def test_prepare_store_all_or_nothing_rollback():
    """If any group's manager cannot prepare, the whole store is rejected
    and already-prepared groups are rolled back — no block may be left
    pending a store that will never be submitted."""
    router = make_router([4, 1])
    # Fill group 1's single block with a write-pending (uncompleted) store:
    # it can neither be freed nor evicted.
    pinned = [key(1, 100)]
    assert router.prepare_store(pinned, _CTX) is not None

    group0 = router._managers[0]
    before_pending = group0._num_write_pending_blocks
    output = router.prepare_store([key(0, 1), key(1, 101)], _CTX)
    assert output is None
    # Group 0's prepared store was rolled back.
    assert group0._num_write_pending_blocks == before_pending
    assert group0.lookup(key(0, 1), _CTX) is not LookupResult.HIT

    # After the pinned store completes, the same store succeeds.
    router.complete_store(pinned, _CTX, success=True)
    assert router.prepare_store([key(0, 1), key(1, 101)], _CTX) is not None


def test_get_stats_capacity_weighted_usage():
    """Usage aggregates over ALL sub-pools' capacity, not per-pool."""
    router = make_router([1, 3])
    # A write-pending (uncompleted) store counts as used; completed stores
    # become evictable cache and leave the "used" gauge.
    output = router.prepare_store([key(0, 1)], _CTX)
    assert output is not None

    stats = router.get_stats()
    assert stats is not None
    reduced = stats.reduce()
    # 1 pending block of 4 total across both pools.
    assert reduced[CPUOffloadingMetrics.CPU_CACHE_USAGE_PERC] == 0.25
    assert reduced[CPUOffloadingMetrics.CPU_CACHE_WRITE_USAGE_PERC] == 0.25


def test_get_stats_emits_zero_stores_skipped():
    """Emission cadence must match the single-pool manager: with a store
    threshold >= 2 the counter is reported every batch, zeros included."""
    router = make_router([4, 4], store_threshold=2)
    stats = router.get_stats()
    assert stats is not None
    assert stats.reduce()[CPUOffloadingMetrics.STORES_SKIPPED] == 0
