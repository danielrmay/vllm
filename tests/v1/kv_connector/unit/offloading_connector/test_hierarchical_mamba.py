# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Offloading-connector coverage for align-mode mamba on a hierarchical
(large-block) pool.

With the mamba block size decoupled from the allocation block size,
``MambaSpec.block_size`` is the small prefix-cache step while one state slot
spans ``large_block_factor`` small blocks. The connector must model the mamba
group at STATE cadence: one offloaded mamba block covers
``block_size * large_block_factor`` tokens, its GPU id is a large id, and the
worker addresses one full ``state_page_size_bytes`` slot per transfer.
"""

import torch

from vllm.v1.kv_cache_interface import (
    FullAttentionSpec,
    KVCacheGroupSpec,
    MambaSpec,
)

from .conftest import request_runner  # noqa: F401
from .utils import generate_store_output

BLOCK = 4
FACTOR = 4
STATE_TOKENS = BLOCK * FACTOR
EOS_TOKEN_ID = 100


def _hybrid_groups() -> list[KVCacheGroupSpec]:
    # Attention page = 2 * 4 * 1 * 1 * 4B = 32B. Mamba state = 32 floats =
    # 128B, so the per-small-block share (128 / FACTOR = 32B) matches the
    # attention page, mirroring the padded byte-exact layout of a real model.
    return [
        KVCacheGroupSpec(
            ["attn"],
            FullAttentionSpec(
                block_size=BLOCK,
                num_kv_heads=1,
                head_size=1,
                dtype=torch.float32,
            ),
        ),
        KVCacheGroupSpec(
            ["mamba"],
            MambaSpec(
                block_size=BLOCK,
                shapes=((32,),),
                dtypes=(torch.float32,),
                mamba_cache_mode="align",
                large_block_factor=FACTOR,
            ),
        ),
    ]


def test_mamba_group_config_uses_state_cadence(request_runner):  # noqa: F811
    """The connector's per-group config must model one offloaded mamba block
    as one full state (block_size * large_block_factor tokens), not as one
    small allocation block."""
    runner = request_runner(
        block_size=BLOCK,
        num_gpu_blocks=32,
        async_scheduling=False,
        kv_cache_groups=_hybrid_groups(),
    )
    mamba_config = runner.connector_scheduler.config.kv_group_configs[1]
    assert mamba_config.tokens_per_block == STATE_TOKENS
    # Mamba depends on a single state: window-1 classification must survive
    # the hierarchical cadence.
    assert mamba_config.sliding_window_size_in_chunks == 1


def test_mamba_stores_under_budget_capped_prefill(request_runner):  # noqa: F811
    """Token budget below the state span (the realistic regime for big
    hybrids: e.g. span 6080 vs budget 2048) must still offload every span
    state.

    With ``max_num_batched_tokens=12 < STATE_TOKENS=16``, no chunk covers a
    whole span, but the state-aligned split steers a chunk end onto every
    span boundary (16, 32, 48, 64), materializing an offloadable state
    there. Each of those states must reach the CPU tier: they are the mamba
    half of the connector hit, and a missing one zeroes the whole hybrid
    lookup for resumes past that span."""
    runner = request_runner(
        block_size=BLOCK,
        num_gpu_blocks=128,
        async_scheduling=False,
        kv_cache_groups=_hybrid_groups(),
        max_num_batched_tokens=12,
    )
    # 4.5 spans of prompt: 72 tokens = 18 attention blocks; span boundaries
    # at 16/32/48/64 = span-END request offsets 3/7/11/15.
    runner.new_request(token_ids=list(range(72)))
    runner.manager.prepare_store.side_effect = lambda keys, req_context: (
        generate_store_output(list(keys))
    )
    # 11 engine steps append 11 tokens -> 83 total: 20 attention chunks and
    # 5 span states (the 5th materializes at the decode-time boundary 80).
    runner.run(
        decoded_tokens=[0] * 10 + [EOS_TOKEN_ID],
        expected_stored=tuple((0, i) for i in range(20))
        + ((1, 3), (1, 7), (1, 11), (1, 15), (1, 19)),
    )


def test_mamba_store_addresses_state_slots(request_runner):  # noqa: F811
    """Driving a request through prefill + finish must offload the mamba
    group as whole state slots addressed by LARGE ids at state cadence.

    A 32-token prompt prefills in a single step, so align materializes only
    the step-end state (the second state span); the first span's slot stays
    null and must be skipped. The attention side stores its full 8-block
    trail. The mamba store lands on the span-END request offset (index 7),
    where align parks the real (large) state block."""
    runner = request_runner(
        block_size=BLOCK,
        num_gpu_blocks=32,
        async_scheduling=False,
        kv_cache_groups=_hybrid_groups(),
    )
    # 2 full states' worth of prompt (32 tokens = 8 attention blocks).
    runner.new_request(token_ids=list(range(2 * STATE_TOKENS)))
    runner.manager.prepare_store.side_effect = lambda keys, req_context: (
        generate_store_output(list(keys))
    )
    # Stores prepared while scheduling the prompt complete during the
    # following run (mirroring test_offloading_connector's cadence).
    runner.run(decoded_tokens=[0])
    runner.run(
        decoded_tokens=[EOS_TOKEN_ID],
        expected_stored=(
            (0, 0),
            (0, 1),
            (0, 2),
            (0, 3),
            (0, 4),
            (0, 5),
            (0, 6),
            (0, 7),
            (1, 7),
        ),
    )


def test_preemption_resets_pending_span_buffer():
    """Preemption must reset BOTH the downsampled block-id list and the
    pending partial-span buffer. The resume re-reports the request's full
    allocation from slot 0, so stale buffer residue phase-shifts every
    later span-end emission onto null padding slots — after the first
    preemption, no mamba state is ever stored again (observed live as
    tail=[0,0,0] with a constant pending offset from span 5 onward, while
    the stride-1 attention group kept storing normally)."""
    from types import SimpleNamespace

    from vllm.distributed.kv_transfer.kv_connector.v1.offloading.scheduler import (
        RequestGroupState,
        RequestOffloadState,
    )

    status = object.__new__(RequestOffloadState)
    status.config = SimpleNamespace(
        kv_group_configs=[SimpleNamespace(small_blocks_per_block=4)]
    )
    status.group_states = (RequestGroupState(),)

    # 1.5 spans ingested: span 1 (real id 7 at its END slot) emitted, 2
    # slots of span 2 left pending.
    status.update_block_id_groups(([0, 0, 0, 7, 0, 0],))
    assert status.group_states[0].block_ids == [7]
    assert len(status.group_states[0].pending_small_block_ids) == 2

    # Preempt, then resume re-reports the full allocation from slot 0.
    status.clear_block_ids()
    status.update_block_id_groups(([0, 0, 0, 7, 0, 0, 0, 9],))

    # Phase-correct span ends: ids 7 and 9. With stale residue this would
    # be [0, 0] (slots 1 and 5 — mid-span nulls).
    assert status.group_states[0].block_ids == [7, 9]
    assert not status.group_states[0].pending_small_block_ids


def test_mismatched_mamba_factors_rejected(request_runner):  # noqa: F811
    """All mamba groups must agree on large_block_factor: the state-aligned
    split steers chunk ends onto ONE span cadence, and a smaller-factor
    group would silently stop producing offloadable states. Today the
    factor comes from a single cache_config value so this cannot happen;
    the scheduler asserts loudly in case that ever changes."""
    import pytest

    groups = _hybrid_groups()
    groups.append(
        KVCacheGroupSpec(
            ["mamba2"],
            MambaSpec(
                block_size=BLOCK,
                shapes=((16,),),
                dtypes=(torch.float32,),
                mamba_cache_mode="align",
                large_block_factor=2,
            ),
        )
    )
    # Two guards can fire depending on construction order: the connector's
    # resolve_mamba_align_size uniform-cadence assert (bare, upstream), or
    # the scheduler's large_block_factor ValueError (ours, covers
    # non-connector deployments too). Either way: loud failure, no silent
    # mis-alignment.
    with pytest.raises((AssertionError, ValueError)):
        request_runner(
            block_size=BLOCK,
            num_gpu_blocks=64,
            async_scheduling=False,
            kv_cache_groups=groups,
        )


def test_hybrid_hit_claim_gated_on_full_state_connectors(request_runner):  # noqa: F811
    """The waiting-queue per-group-MAX hit shortcut may only fire for
    connectors that transfer the mamba state for the full claimed prefix
    (NIXL). For any other connector the scheduler must claim the RECONCILED
    hybrid hit: with the mamba states evicted, that is ZERO — claiming the
    attention-only hit would resume with a null mamba state and silently
    poison every state cached downstream."""
    runner = request_runner(
        block_size=BLOCK,
        num_gpu_blocks=128,
        async_scheduling=False,
        kv_cache_groups=_hybrid_groups(),
    )
    prefix = list(range(20))
    runner.new_request(token_ids=prefix)
    runner.manager.prepare_store.side_effect = lambda keys, req_context: (
        generate_store_output([])
    )
    runner.run(decoded_tokens=[0])
    runner.run(decoded_tokens=[EOS_TOKEN_ID])

    # Surgically evict ONLY the mamba states (the NIXL-motivating scenario:
    # attention prefix survives, states gone).
    pool = runner.scheduler.kv_cache_manager.coordinator.block_pool
    for meta in pool.large_block_metas:
        large = meta.large_block
        if (
            large.block_hash is not None
            or pool._hash_index_key(large) in pool.cached_block_hashes_by_block
        ):
            pool._remove_cached_block_hashes(large)

    # Same prefix + continuation; observe the schedule-time claim through
    # the number of NEW tokens scheduled.
    runner.new_request(token_ids=prefix + list(range(100, 116)))
    output = runner.scheduler.schedule()
    rid = str(runner.req_id)
    # Default (offloading connector): the reconciled hybrid hit is ZERO —
    # attention alone may not be claimed. Post-schedule num_computed equals
    # hit + scheduled, so hit == computed - scheduled == 0. (The 20-token
    # chunk is the split's shared-prefix junction stop, not a cache claim.)
    req = runner.scheduler.requests[rid]
    scheduled = output.num_scheduled_tokens[rid]
    assert req.num_computed_tokens - scheduled == 0
    assert scheduled == 20


def test_hybrid_hit_claim_max_branch_for_full_state_connectors(request_runner):  # noqa: F811
    """Counterpart: with _connector_transfers_full_mamba_state set (as the
    NIXL connector configuration does), the scheduler claims the per-group
    MAX (the attention hit) and only the tail is scheduled — sound only
    because such connectors ship the mamba state unconditionally."""
    runner = request_runner(
        block_size=BLOCK,
        num_gpu_blocks=128,
        async_scheduling=False,
        kv_cache_groups=_hybrid_groups(),
    )
    prefix = list(range(20))
    runner.new_request(token_ids=prefix)
    runner.manager.prepare_store.side_effect = lambda keys, req_context: (
        generate_store_output([])
    )
    runner.run(decoded_tokens=[0])
    runner.run(decoded_tokens=[EOS_TOKEN_ID])

    pool = runner.scheduler.kv_cache_manager.coordinator.block_pool
    for meta in pool.large_block_metas:
        large = meta.large_block
        if (
            large.block_hash is not None
            or pool._hash_index_key(large) in pool.cached_block_hashes_by_block
        ):
            pool._remove_cached_block_hashes(large)

    # The flag<->connector-name mapping is pinned separately
    # (test_connector_transfers_full_mamba_state_flag); flip it here to
    # exercise the gated branch itself.
    runner.scheduler._connector_transfers_full_mamba_state = True
    runner.new_request(token_ids=prefix + list(range(100, 116)))
    output = runner.scheduler.schedule()
    rid = str(runner.req_id)
    # Attention hit (20) claimed (hit == computed - scheduled); the resumed
    # chunk re-aligns to the next span boundary (32) -> 12 new tokens.
    req = runner.scheduler.requests[rid]
    scheduled = output.num_scheduled_tokens[rid]
    assert req.num_computed_tokens - scheduled == 20
    assert scheduled == 12


def test_mixed_fine_local_and_span_external_hit(request_runner):  # noqa: F811
    """A fine-grained local GPU hit (16-token grid) combined with a
    span-cadence external CPU hit must compose: the connector's claim is
    rounded down to the state span (resolve_mamba_align_size), so the
    total cached count is span-aligned and update_state_after_alloc's
    span-count assert holds. Loads must cover exactly the gap between the
    fine local hit and the span boundary."""
    runner = request_runner(
        block_size=BLOCK,
        num_gpu_blocks=128,
        async_scheduling=False,
        kv_cache_groups=_hybrid_groups(),
    )
    prefix36 = list(range(36))

    # Phase 1: store the first two spans (32 tokens) to the CPU tier:
    # attention chunks 0-7 plus the STEP-END state (span 2): a single-chunk
    # prefill materializes only the final state; span 1's slot stays null.
    runner.new_request(token_ids=prefix36[:32])
    runner.manager.prepare_store.side_effect = lambda keys, req_context: (
        generate_store_output(list(keys))
    )
    runner.run(decoded_tokens=[0])
    runner.run(
        decoded_tokens=[EOS_TOKEN_ID],
        expected_stored=tuple((0, i) for i in range(8)) + ((1, 7),),
    )

    # Phase 2: surgically drop every GPU cache entry (reset_prefix_cache
    # declines while hierarchical metas are partial), then re-cache only a
    # FINE, non-span-aligned prefix (8 of 16 tokens).
    pool = runner.scheduler.kv_cache_manager.coordinator.block_pool
    for block in pool.blocks:
        if block.block_hash is not None:
            pool._remove_cached_block_hashes(block)
    for meta in pool.large_block_metas:
        large = meta.large_block
        if (
            large.block_hash is not None
            or pool._hash_index_key(large) in pool.cached_block_hashes_by_block
        ):
            pool._remove_cached_block_hashes(large)
    runner.manager.prepare_store.side_effect = lambda keys, req_context: (
        generate_store_output([])
    )
    runner.new_request(token_ids=prefix36[:8])
    runner.run(decoded_tokens=[EOS_TOKEN_ID])

    # CPU tier claims exactly one full-attention alignment's worth: 4
    # attention chunks (16 tokens) and 1 mamba span state.
    # Every key hits in the CPU tier; the lookup's sliding-window -1 and
    # span round-down clip the usable claim to 16 tokens.
    from vllm.v1.kv_offload.base import LookupResult

    runner.manager.lookup.side_effect = lambda key, req_context: LookupResult.HIT
    runner.new_request(token_ids=prefix36)
    # local hit = 8 (FINE, non-span-aligned); claim = 32 (span-rounded) ->
    # external = 24: attention chunks 2-7 load, and mamba - window-1 - loads
    # ONLY the claim-boundary state, into its span-END slot (offset 7).
    # This exercises update_state_after_alloc's span-count assert with a
    # fine local + span external composition.
    runner.run(
        decoded_tokens=[0, EOS_TOKEN_ID],
        expected_loaded=tuple((0, i) for i in range(2, 8)) + ((1, 7),),
    )


def test_multiconnector_wrapped_offloading(request_runner):  # noqa: F811
    """The state-cadence machinery must engage when the offloading connector
    is nested inside MultiConnector: the split gate reads sub-connector
    names, and stores land at span cadence exactly as unwrapped."""
    runner = request_runner(
        block_size=BLOCK,
        num_gpu_blocks=32,
        async_scheduling=False,
        kv_cache_groups=_hybrid_groups(),
        wrap_multi_connector=True,
    )
    assert runner.scheduler.mamba_split_state_aligned

    runner.new_request(token_ids=list(range(2 * STATE_TOKENS)))
    runner.manager.prepare_store.side_effect = lambda keys, req_context: (
        generate_store_output(list(keys))
    )
    runner.run(decoded_tokens=[0])
    runner.run(
        decoded_tokens=[EOS_TOKEN_ID],
        expected_stored=(
            (0, 0),
            (0, 1),
            (0, 2),
            (0, 3),
            (0, 4),
            (0, 5),
            (0, 6),
            (0, 7),
            (1, 7),
        ),
    )
