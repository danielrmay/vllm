# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Direct unit coverage for the worker-side CoW copy. This path only fires
on partial-tail prefix hits, which ordinary resume workloads (and the e2e
suite) do not produce — so it must be pinned at unit level."""

import pytest
import torch

from vllm.v1.worker.utils import copy_kv_cache_blocks_inplace

pytestmark = pytest.mark.cpu_test

PAGE = 32  # bytes per small page
NUM_BLOCKS = 8


def _block_major(num_rows: int, fill_base: int) -> torch.Tensor:
    """A block-major uint8 storage where row i is filled with fill_base+i."""
    rows = torch.arange(num_rows, dtype=torch.uint8) + fill_base
    return rows.repeat_interleave(PAGE).clone()


def test_copies_apply_to_full_and_short_storages():
    # Attention layer: full-size storage (one row per pool block).
    attention = _block_major(NUM_BLOCKS, fill_base=10)
    # Hierarchical mamba layer: SHORT storage — state slots cover only the
    # factor-aligned prefix (4 of 8 pool rows here). Lists of tensors
    # aliasing one storage are how mamba layers arrive.
    short = _block_major(4, fill_base=100)
    mamba_views = [short[: 2 * PAGE], short]

    copy_kv_cache_blocks_inplace(
        kv_caches=[attention, mamba_views],
        num_blocks=NUM_BLOCKS,
        page_size=PAGE,
        kv_cache_block_copies=[(1, 5), (2, 3)],
    )

    att = attention.view(NUM_BLOCKS, PAGE)
    # (1 -> 5) and (2 -> 3) applied to the full storage.
    assert (att[5] == att[1]).all() and att[5][0].item() == 11
    assert (att[3] == att[2]).all() and att[3][0].item() == 12
    # Untouched rows keep their fill.
    assert att[0][0].item() == 10 and att[7][0].item() == 17

    sh = short.view(4, PAGE)
    # (2 -> 3) lands inside the short storage's 4 rows: applied.
    assert (sh[3] == sh[2]).all() and sh[3][0].item() == 102
    # (1 -> 5) aims past its range: skipped, nothing clobbered.
    assert sh[1][0].item() == 101


def test_out_of_range_copies_are_masked_per_copy():
    """A batch mixing in-range and out-of-range copies applies the in-range
    subset to the SHORT storage (per-copy masking, not whole-batch skip)
    while the full storage applies everything — pin that shape so a future
    refactor doesn't silently change it."""
    attention = _block_major(NUM_BLOCKS, fill_base=10)
    short = _block_major(4, fill_base=100)

    copy_kv_cache_blocks_inplace(
        kv_caches=[attention, [short]],
        num_blocks=NUM_BLOCKS,
        page_size=PAGE,
        kv_cache_block_copies=[(0, 1), (6, 7)],
    )

    att = attention.view(NUM_BLOCKS, PAGE)
    assert att[1][0].item() == 10 and att[7][0].item() == 16
    sh = short.view(4, PAGE)
    # In-range (0, 1) applied; out-of-range (6, 7) skipped for this storage.
    assert sh[1][0].item() == 100
    assert sh[2][0].item() == 102 and sh[3][0].item() == 103


def test_empty_copy_list_is_noop():
    attention = _block_major(NUM_BLOCKS, fill_base=10)
    before = attention.clone()
    copy_kv_cache_blocks_inplace(
        kv_caches=[attention],
        num_blocks=NUM_BLOCKS,
        page_size=PAGE,
        kv_cache_block_copies=[],
    )
    assert (attention == before).all()


def test_packed_storage_copies_whole_block_stride():
    """A packed storage (k pages per block, e.g. cross-layer blocks) must be
    addressed at its own block stride: block id i owns bytes
    [i*k*PAGE, (i+1)*k*PAGE). Viewing it at the uniform page size would copy
    the i-th PAGE instead of the i-th block — silent corruption. Regression
    for the per-storage stride derivation (legacy behavior restored)."""
    k = 2
    packed = (
        (torch.arange(NUM_BLOCKS, dtype=torch.uint8) + 50)
        .repeat_interleave(k * PAGE)
        .clone()
    )
    # Make the two pages of each block distinguishable.
    view = packed.view(NUM_BLOCKS, k, PAGE)
    view[:, 1, :] += 100

    copy_kv_cache_blocks_inplace(
        kv_caches=[packed],
        num_blocks=NUM_BLOCKS,
        page_size=PAGE,
        kv_cache_block_copies=[(1, 5)],
    )

    v = packed.view(NUM_BLOCKS, k, PAGE)
    # BOTH pages of block 1 landed in block 5 (block-stride addressing).
    assert v[5][0][0].item() == 51 and v[5][1][0].item() == 151
    # Neighbors untouched.
    assert v[4][0][0].item() == 54 and v[6][0][0].item() == 56


def test_oversized_non_divisible_storage_refused():
    """A storage larger than num_blocks x page_size that does not divide
    evenly into blocks is an unknown layout; page-granularity copies would
    miscopy it (the packed-layout bug via a different door), so it must
    fail loudly instead of silently reclassifying as short."""
    import pytest

    # 8 blocks x 32B pages = 256B; 300B is oversized and non-divisible.
    weird = torch.zeros(300, dtype=torch.uint8)
    with pytest.raises(ValueError, match="unknown layout"):
        copy_kv_cache_blocks_inplace(
            kv_caches=[weird],
            num_blocks=NUM_BLOCKS,
            page_size=PAGE,
            kv_cache_block_copies=[(0, 1)],
        )


def test_resolve_cow_page_unwraps_uniform_type_groups():
    """A UniformTypeKVCacheSpecs group reports the SUM of member pages as
    its page_size_bytes; CoW addressing must use the per-LAYER page (or
    refuse when members disagree), never the aggregate."""
    from types import SimpleNamespace

    from vllm.v1.kv_cache_interface import (
        FullAttentionSpec,
        UniformTypeKVCacheSpecs,
    )
    from vllm.v1.worker.utils import resolve_uniform_cow_page_size

    def attn(num_kv_heads):
        return FullAttentionSpec(
            block_size=16,
            num_kv_heads=num_kv_heads,
            head_size=8,
            dtype=torch.float32,
        )

    same = UniformTypeKVCacheSpecs(
        block_size=16, kv_cache_specs={"a": attn(1), "b": attn(1)}
    )
    assert same.page_size_bytes == 2 * attn(1).page_size_bytes  # the trap
    groups = [SimpleNamespace(kv_cache_spec=same)]
    assert resolve_uniform_cow_page_size(groups) == attn(1).page_size_bytes

    mixed = UniformTypeKVCacheSpecs(
        block_size=16, kv_cache_specs={"a": attn(1), "b": attn(2)}
    )
    groups = [SimpleNamespace(kv_cache_spec=mixed)]
    assert resolve_uniform_cow_page_size(groups) is None
