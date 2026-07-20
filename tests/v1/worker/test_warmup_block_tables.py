# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""CPU unit coverage for the V2 warmup block-table builder — logic that has
been wrong once (per-group counters aliased rows across small-granularity
groups sharing one id space) and is otherwise only exercised implicitly by
GPU warmup at startup."""

import pytest
import torch

from vllm.v1.kv_cache_interface import FullAttentionSpec, MambaSpec
from vllm.v1.worker.gpu.warmup import (
    _group_table_shape,
    _make_group_table_builder,
)

pytestmark = pytest.mark.cpu_test

BLOCK_SIZE = 16
FACTOR = 4
NUM_BLOCKS = 64


def _attn():
    return FullAttentionSpec(
        block_size=BLOCK_SIZE, num_kv_heads=1, head_size=8, dtype=torch.float32
    )


def _mamba(num_speculative_blocks=1):
    return MambaSpec(
        block_size=BLOCK_SIZE,
        shapes=((1,), (1,)),
        dtypes=(torch.float32, torch.float32),
        mamba_cache_mode="align",
        large_block_factor=FACTOR,
        num_speculative_blocks=num_speculative_blocks,
    )


def _builder(specs):
    limits = [
        NUM_BLOCKS // (s.large_block_factor if isinstance(s, MambaSpec) else 1)
        for s in specs
    ]
    return _make_group_table_builder(specs, limits)


def test_small_groups_share_one_globally_distinct_id_space():
    """Two factor-1 groups must draw sequentially from ONE counter: their
    layers can share backing tensors, so per-group counters restarting at
    1 would alias warmup rows (the round-13 regression)."""
    specs = [_attn(), _attn()]
    build_table, build_delta = _builder(specs)

    tables = [build_table(g, 3 * BLOCK_SIZE) for g in (0, 1)]
    ids_a, ids_b = set(tables[0]), set(tables[1])
    assert not (ids_a & ids_b), "small groups drew overlapping ids"
    assert sorted(ids_a | ids_b) == list(range(1, 7))  # one shared sequence

    # Delta rows continue the same shared sequence.
    delta = build_delta(0, 1)
    assert delta == [7]


def test_hierarchical_mamba_ids_stay_in_state_range():
    specs = [_attn(), _mamba()]
    build_table, _ = _builder(specs)

    # Sparse align table: null-padded, real state id at the LAST token-range
    # row, speculative snapshot slots appended.
    rows, real = _group_table_shape(specs[1], 5 * BLOCK_SIZE)
    assert rows == 5 and real == 2  # 1 state slot + 1 speculative
    table = build_table(1, 5 * BLOCK_SIZE)
    assert len(table) == rows + 1  # token rows + appended snapshot slot
    assert table[:4] == [0, 0, 0, 0]
    real_ids = [i for i in table if i != 0]
    assert len(real_ids) == 2 and table[4] == real_ids[0]
    assert all(0 < i < NUM_BLOCKS // FACTOR for i in real_ids)

    # The attention group's counter is unaffected by mamba draws.
    attn_table = build_table(0, 2 * BLOCK_SIZE)
    assert attn_table == [1, 2]


def test_take_ids_overrun_raises():
    specs = [_attn(), _mamba()]
    build_table, _ = _builder(specs)
    # The mamba group's range is NUM_BLOCKS // FACTOR = 16 ids (1..15
    # usable). Consuming 2 real ids per request: request 8 overruns.
    with pytest.raises(ValueError, match="block-id space"):
        for _ in range(8):
            build_table(1, 5 * BLOCK_SIZE)
