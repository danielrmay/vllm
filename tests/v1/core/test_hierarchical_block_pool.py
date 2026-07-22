# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for the hierarchical (large-block) BlockPool used by hybrid models
with mamba large blocks spanning N attention-sized small blocks."""

from types import SimpleNamespace

from vllm.v1.core.block_pool import BlockPool
from vllm.v1.core.kv_cache_manager import KVCacheManager
from vllm.v1.core.kv_cache_utils import (
    BlockHash,
    make_block_hash_with_group_id,
)

N = 4
NUM_SMALL = 32
NUM_LARGE = NUM_SMALL // N  # 8, meta 0 is reserved for the null block


def make_pool() -> BlockPool:
    return BlockPool(
        num_gpu_blocks=NUM_SMALL,
        enable_caching=True,
        hash_block_size=2,
        large_block_factor=N,
    )


def cache_block(pool: BlockPool, block, seed: str) -> None:
    """Register ``block`` in the prefix cache under a deterministic hash."""
    key = make_block_hash_with_group_id(BlockHash(seed.encode()), 0)
    block.set_block_hash(key, num_tokens=None)
    pool.cached_block_hash_to_block.insert(key, block)


def lookup(pool: BlockPool, seed: str):
    key = make_block_hash_with_group_id(BlockHash(seed.encode()), 0)
    return pool.cached_block_hash_to_block.get_one_block(key)


def test_init_hierarchical_pool():
    pool = make_pool()
    assert len(pool.large_block_metas) == NUM_LARGE
    # Meta 0 hosts the null block and never enters a free queue.
    assert pool.null_block is pool.large_block_metas[0].small_blocks[0]
    assert pool.null_block.is_null
    assert pool.get_num_free_large_blocks() == NUM_LARGE - 1
    assert pool.get_num_free_blocks() == (NUM_LARGE - 1) * N
    # Small blocks of meta L carry consecutive ids [L*N, (L+1)*N).
    for L, meta in enumerate(pool.large_block_metas):
        assert meta.large_block.block_id == L
        assert [b.block_id for b in meta.small_blocks] == list(
            range(L * N, (L + 1) * N)
        )


def test_large_alloc_free_roundtrip():
    pool = make_pool()
    larges = pool.get_new_blocks(2, large_block=True)
    assert pool.get_num_free_large_blocks() == NUM_LARGE - 3
    for blk in larges:
        meta = pool.large_block_metas[blk.block_id]
        assert meta.large_block is blk
        assert meta.num_small_in_use == N
        assert meta.next_small_idx == N
    pool.free_blocks(larges)
    assert pool.get_num_free_large_blocks() == NUM_LARGE - 1
    for blk in larges:
        meta = pool.large_block_metas[blk.block_id]
        assert meta.num_small_in_use == 0
        assert meta.next_small_idx == 0


def test_small_dispense_continues_in_parent():
    pool = make_pool()
    first = pool.get_new_blocks(3)
    parent = pool.large_block_metas[first[0].block_id // N]
    assert parent.next_small_idx == 3
    # Continuation stays inside the same parent for the residual slot ...
    more = pool.get_new_blocks(2, last_hit_block_id=first[-1].block_id)
    assert more[0].block_id // N == first[0].block_id // N
    assert parent.next_small_idx == N
    # ... and overflows into a freshly opened meta.
    assert more[1].block_id // N != first[0].block_id // N


def test_small_free_recycles_meta():
    pool = make_pool()
    smalls = pool.get_new_blocks(N)
    meta = pool.large_block_metas[smalls[0].block_id // N]
    free_large_before = pool.get_num_free_large_blocks()
    pool.free_blocks(smalls)
    assert meta.num_small_in_use == 0
    assert meta.next_small_idx == 0
    assert pool.get_num_free_large_blocks() == free_large_before + 1


def test_partial_meta_recycles_without_full_cursor():
    """A request that finished before consuming its whole meta must not leak
    the meta: in-use hitting zero recycles it even with a partial cursor."""
    pool = make_pool()
    smalls = pool.get_new_blocks(2)
    meta = pool.large_block_metas[smalls[0].block_id // N]
    assert 0 < meta.next_small_idx < N
    free_large_before = pool.get_num_free_large_blocks()
    pool.free_blocks(smalls)
    assert meta.next_small_idx == 0
    assert pool.get_num_free_large_blocks() == free_large_before + 1


def test_repurpose_as_large_evicts_stale_small_hashes():
    """Cross-granularity stale-hash eviction, small->large direction: a meta
    whose small blocks still carry prefix-cache entries is repurposed as one
    mamba state slot; the stale small hashes must be evicted."""
    pool = make_pool()
    smalls = pool.get_new_blocks(N)
    cache_block(pool, smalls[0], "stale-small")
    pool.free_blocks(smalls)
    assert lookup(pool, "stale-small") is not None  # lingers while free

    # Drain free-large until the recycled meta is dispensed as a LARGE block.
    target_meta = pool.large_block_metas[smalls[0].block_id // N]
    seen = []
    while pool.get_num_free_large_blocks() > 0:
        blk = pool.get_new_blocks(1, large_block=True)[0]
        seen.append(blk)
        if blk is target_meta.large_block:
            break
    assert seen[-1] is target_meta.large_block
    assert lookup(pool, "stale-small") is None


def test_repurpose_as_small_evicts_stale_large_hash():
    """Cross-granularity stale-hash eviction, large->small direction: a freed
    mamba state slot still carrying its hash is re-dispensed as attention
    small blocks; the stale large hash must be evicted."""
    pool = make_pool()
    large = pool.get_new_blocks(1, large_block=True)[0]
    cache_block(pool, large, "stale-large")
    pool.free_blocks([large])
    assert lookup(pool, "stale-large") is not None  # lingers while free

    target_meta = pool.large_block_metas[large.block_id]
    got_target = False
    while pool.get_num_free_large_blocks() > 0:
        small = pool.get_new_blocks(1)[0]
        if small.block_id // N == large.block_id:
            got_target = True
            break
    assert got_target
    assert lookup(pool, "stale-large") is None
    assert target_meta.large_block.block_hash is None


def test_free_mixed_list_identity_classification():
    """free_blocks must classify each block by IDENTITY, never by numeric
    id value: call sites can free mixed lists whose ids collide (a large
    block with id L and a small block with block_id == L)."""
    pool = make_pool()
    # Hold the large block with id N (== the id of meta 1's first small
    # block), then release the earlier metas so their smalls get dispensed:
    # this manufactures the numeric id collision.
    larges = pool.get_new_blocks(N, large_block=True)
    large = next(blk for blk in larges if blk.block_id == N)
    pool.free_blocks([blk for blk in larges if blk is not large])
    small = None
    while pool.get_num_free_blocks() > 0:
        blk = pool.get_new_blocks(1)[0]
        if blk.block_id == large.block_id:
            small = blk
            break
    assert small is not None and small is not large

    small_meta = pool.large_block_metas[small.block_id // N]
    in_use_before = small_meta.num_small_in_use
    large_meta = pool.large_block_metas[large.block_id]

    # Free the mixed list in one call; granularity is per-block.
    pool.free_blocks([small, large])

    # The small free must have decremented ITS parent's accounting ...
    assert small_meta.num_small_in_use == in_use_before - 1
    # ... and the large free must have reset ITS meta wholesale.
    assert large_meta.num_small_in_use == 0
    assert large_meta.next_small_idx == 0


def test_touch_reclaims_freed_small_from_recycled_meta():
    pool = make_pool()
    smalls = pool.get_new_blocks(N)
    cache_block(pool, smalls[1], "hit-me")
    meta = pool.large_block_metas[smalls[0].block_id // N]
    pool.free_blocks(smalls)
    assert meta.num_small_in_use == 0  # recycled into free-large

    hit = lookup(pool, "hit-me")
    assert hit is smalls[1]
    free_large_before = pool.get_num_free_large_blocks()
    pool.touch([hit])
    # The parent meta was pulled back out of the free-large queue and the
    # cursor advanced past the re-touched slot.
    assert pool.get_num_free_large_blocks() == free_large_before - 1
    assert meta.num_small_in_use == 1
    assert meta.next_small_idx >= (hit.block_id % N) + 1
    assert hit.ref_cnt == 1


def test_reset_prefix_cache_clears_large_hashes():
    pool = make_pool()
    large = pool.get_new_blocks(1, large_block=True)[0]
    cache_block(pool, large, "large-hash")
    pool.free_blocks([large])
    assert pool.reset_prefix_cache()
    assert large.block_hash is None
    assert lookup(pool, "large-hash") is None


def test_reset_prefix_cache_refuses_when_in_use():
    pool = make_pool()
    large = pool.get_new_blocks(1, large_block=True)[0]
    assert not pool.reset_prefix_cache()
    pool.free_blocks([large])
    assert pool.reset_prefix_cache()


def test_cow_copies_expand_large_ids_to_small_pages():
    """take_kv_cache_block_copies must emit small-page-major copies: the
    worker's copy kernel views the backing storage as (num_small, page), so a
    large-block manager's CoW pair (large ids) expands into
    large_block_factor consecutive small-page copies."""
    pool = make_pool()
    src = pool.get_new_blocks(1, large_block=True)[0]
    dst = pool.get_new_blocks(1, large_block=True)[0]

    mamba_mgr = SimpleNamespace(
        is_large_block=True,
        take_pending_cow_copies=lambda: [(src, dst)],
    )
    small_src, small_dst = pool.get_new_blocks(2)
    attn_mgr = SimpleNamespace(
        is_large_block=False,
        take_pending_cow_copies=lambda: [(small_src, small_dst)],
    )
    manager = KVCacheManager.__new__(KVCacheManager)
    manager.coordinator = SimpleNamespace(
        block_pool=pool, single_type_managers=[mamba_mgr, attn_mgr]
    )

    copies, retained = manager.take_kv_cache_block_copies()

    expected_large = [(src.block_id * N + j, dst.block_id * N + j) for j in range(N)]
    expected = expected_large + [(small_src.block_id, small_dst.block_id)]
    assert [(c.src_block_id, c.dst_block_id) for c in copies] == expected
    assert retained == [src, dst, small_src, small_dst]


def test_partial_hash_index_immune_to_small_id_collision():
    """Large and small block ids share the numeric range [0, num_large), so
    the partial-hash side index must be keyed by granularity, not bare id:
    evicting/freeing a SMALL block must never drop an unrelated LARGE
    block's fine-grained (partial-hash) entries, and move_block_hashes'
    dst-is-clean assert must not false-fire on the numeric twin."""
    pool = BlockPool(
        num_gpu_blocks=32,
        enable_caching=True,
        hash_block_size=4,
        large_block_factor=4,
    )
    # A large block with a primary hash plus one partial (secondary) entry.
    large = pool.large_block_metas[1].large_block
    h_primary = make_block_hash_with_group_id(BlockHash(b"collision-primary"), 0)
    h_partial = make_block_hash_with_group_id(BlockHash(b"collision-partial"), 0)
    pool._insert_block_hash(h_primary, large, num_tokens=16)
    pool._insert_block_hash(h_partial, large, num_tokens=4)
    assert pool.cached_block_hash_to_block.get_one_block(h_partial) is large

    # The small block with the SAME numeric id as the large.
    small_twin = pool.blocks[large.block_id]
    assert small_twin is not large

    # Freeing/evicting the small twin must not disturb the large's entries.
    pool._remove_cached_block_hashes(small_twin)
    assert pool.cached_block_hash_to_block.get_one_block(h_partial) is large, (
        "small-block eviction dropped an unrelated large block's "
        "partial-hash entries (bare-id keying collision)"
    )
    assert pool.cached_block_hash_to_block.get_one_block(h_primary) is large


def test_coordinator_refuses_multiple_small_granularity_groups():
    """Admission sums small-unit counts globally, but each small-granularity
    group opens its OWN metas at dispense (one group cannot use another's
    partial-meta residual). Two small groups can therefore pass admission
    and still fail allocation near pool exhaustion — refuse loudly at init
    until per-group meta fragmentation is accounted for. One small group +
    hierarchical mamba groups (every in-tree hybrid) must construct."""
    import pytest
    import torch

    from vllm.v1.core.kv_cache_coordinator import get_kv_cache_coordinator
    from vllm.v1.kv_cache_interface import (
        FullAttentionSpec,
        KVCacheConfig,
        KVCacheGroupSpec,
        MambaSpec,
        SlidingWindowSpec,
    )

    def spec_args():
        return dict(num_kv_heads=1, head_size=1, dtype=torch.float32)

    full = FullAttentionSpec(block_size=16, **spec_args())
    swa = SlidingWindowSpec(block_size=16, sliding_window=64, **spec_args())
    mamba = MambaSpec(
        block_size=16,
        shapes=((1,), (1,)),
        dtypes=(torch.float32,),
        mamba_cache_mode="align",
        large_block_factor=4,
    )

    def coordinator(groups):
        return get_kv_cache_coordinator(
            kv_cache_config=KVCacheConfig(
                num_blocks=64,
                kv_cache_tensors=[],
                kv_cache_groups=[
                    KVCacheGroupSpec(layer_names=[f"layer.{i}"], kv_cache_spec=s)
                    for i, s in enumerate(groups)
                ],
            ),
            max_model_len=1024,
            max_in_flight_tokens=0,
            use_eagle=False,
            enable_caching=True,
            enable_kv_cache_events=False,
            dcp_world_size=1,
            pcp_world_size=1,
            scheduler_block_size=16,
            hash_block_size=16,
        )

    # One small-granularity group + hierarchical mamba: fine.
    coordinator([full, mamba])

    # Two small-granularity groups + hierarchical mamba: refuse at init.
    with pytest.raises(NotImplementedError, match="small-granularity"):
        coordinator([full, swa, mamba])
