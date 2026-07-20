# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Per-cost-class CPU pool sizing guards."""

from types import SimpleNamespace

import pytest

from vllm.v1.kv_offload.cpu.spec import CPUOffloadingSpec


def _config(cpu_bytes: int) -> SimpleNamespace:
    # Two cost classes: cheap attention chunks vs expensive mamba states.
    groups = (
        SimpleNamespace(kv_bytes_per_block=1024, tokens_per_block=16),
        SimpleNamespace(kv_bytes_per_block=1 << 20, tokens_per_block=256),
    )
    return SimpleNamespace(
        groups=groups,
        worker_kv_bytes_per_block=1024,
        enable_kv_cache_events=False,
        extra_config={"cpu_bytes_to_use": cpu_bytes},
        engine_id="test",
        model=SimpleNamespace(),
        cache=SimpleNamespace(block_size=16, tokens_per_hash=16, blocks_per_chunk=1),
        parallel=SimpleNamespace(world_size=1),
    )


def test_zero_capacity_cost_class_rejected():
    """A cost class whose byte-rate share affords < 1 chunk would make its
    prepare_store return None; the router's all-or-nothing contract then
    rejects EVERY store, silently disabling the CPU tier. Must fail loudly
    at init with the sizing math instead."""
    with pytest.raises(ValueError, match="kv_offloading_size"):
        CPUOffloadingSpec(_config(cpu_bytes=1 << 20))


def test_adequate_budget_accepted():
    spec = CPUOffloadingSpec(_config(cpu_bytes=1 << 30))
    assert spec.num_blocks_per_group is not None
    assert all(n >= 1 for n in spec.num_blocks_per_group)


def test_error_suggested_budget_is_sufficient():
    """The suggested kv_offloading_size must size EVERY cost class for at
    least one chunk, not just the class that failed first."""
    import re

    with pytest.raises(ValueError) as excinfo:
        CPUOffloadingSpec(_config(cpu_bytes=1 << 20))
    match = re.search(r"\((\d+) bytes;", str(excinfo.value))
    assert match is not None
    suggested = int(match.group(1))
    spec = CPUOffloadingSpec(_config(cpu_bytes=suggested))
    assert spec.num_blocks_per_group is not None
    assert all(n >= 1 for n in spec.num_blocks_per_group)


def test_tiering_spec_rejects_heterogeneous_costs():
    """TieringOffloadingSpec builds its regions on the uniform-pool fields,
    which the per-cost-class branch leaves unset; composing them would
    create a zero-byte region. Must refuse loudly at init."""
    from vllm.v1.kv_offload.tiering.spec import TieringOffloadingSpec

    config = _config(cpu_bytes=1 << 30)
    config.kv_events_config = SimpleNamespace(self_describing_kv_events=False)
    with pytest.raises(NotImplementedError, match="heterogeneous"):
        TieringOffloadingSpec(config)
