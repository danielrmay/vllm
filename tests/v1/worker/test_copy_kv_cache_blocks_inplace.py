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
        kv_caches=[attention], page_size=PAGE, kv_cache_block_copies=[]
    )
    assert (attention == before).all()
