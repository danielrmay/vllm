# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from collections.abc import Callable
from contextlib import AbstractContextManager, nullcontext
from typing import Any

import numpy as np
import torch

from vllm import PoolingParams, SamplingParams
from vllm.logger import init_logger
from vllm.multimodal.inputs import MultiModalFeatureSpec, PlaceholderRange
from vllm.utils.math_utils import cdiv
from vllm.v1.core.sched.output import (
    CachedRequestData,
    GrammarOutput,
    NewRequestData,
    SchedulerOutput,
)
from vllm.v1.kv_cache_interface import CrossAttentionSpec, MambaSpec
from vllm.v1.request import Request
from vllm.v1.worker.gpu.model_runner import GPUModelRunner

logger = init_logger(__name__)


def _group_factor(spec: Any) -> int:
    return spec.large_block_factor if isinstance(spec, MambaSpec) else 1


def _is_sparse_align_mamba(spec: Any) -> bool:
    return (
        isinstance(spec, MambaSpec)
        and spec.mamba_cache_mode == "align"
        and spec.large_block_factor > 1
    )


def _group_id_limits(model_runner: GPUModelRunner, kv_cache_groups: Any) -> list[int]:
    """Highest addressable block id (exclusive) per KV group: attention
    groups address [0, num_blocks) small ids, a hierarchical mamba group's
    state view has only num_blocks // large_block_factor rows."""
    num_blocks = model_runner.kv_cache_config.num_blocks
    return [num_blocks // _group_factor(g.kv_cache_spec) for g in kv_cache_groups]


def _group_table_shape(spec: Any, num_tokens: int) -> tuple[int, int]:
    """(table_rows, real_ids_consumed) for one request's warmup block table.

    Align-mode hierarchical mamba tables are FINE-granularity and sparse at
    runtime: one row per ``spec.block_size`` tokens, null everywhere except
    the state-slot position (plus appended speculative snapshot slots) —
    see ``MambaManager`` align allocation and its shape test. A dense
    span-granularity table would leave the backend's row lookups in the
    zero padding, resolving every warmup request to the null state slot.
    Other groups use dense tables: one real id per allocation unit
    (``block_size`` x factor tokens).
    """
    if _is_sparse_align_mamba(spec):
        rows = cdiv(num_tokens, spec.block_size)
        real = 1 + spec.num_speculative_blocks
        return rows, real
    rows = cdiv(num_tokens, spec.block_size * _group_factor(spec))
    if isinstance(spec, MambaSpec) and spec.mamba_cache_mode == "align":
        # Flat align (factor == 1): snapshot slots append as real rows.
        rows += spec.num_speculative_blocks
    return rows, rows


def _make_group_table_builder(kv_cache_specs: list[Any], group_id_limits: list[int]):
    """Per-group warmup block-table builder (0 is the null block).

    Small-granularity (factor == 1) groups share ONE sequential counter:
    their ids must be globally distinct across groups, exactly like the
    real scheduler's single global pool — different groups' layers can
    share backing tensors, so per-group counters restarting at 1 would
    alias rows and let later layers overwrite earlier layers' warmup KV.
    Hierarchical mamba groups (factor > 1) get per-group counters bounded
    by their smaller state range (num_blocks // factor). Their slots DO
    overlay small ids in shared buffers (each KVCacheTensor is shared by
    one layer per group), so cross-granularity aliasing does occur during
    warmup — harmless because warmup values are never read for
    correctness: warmup blocks never enter the pool or prefix cache, and
    serving writes before reading. Only equal-granularity groups sharing
    one id space need globally distinct ids, to keep warmup
    shape-faithful. Sparse align-mamba tables get nulls at non-state positions,
    a real state-slot id at the last token-range row, and the speculative
    snapshot slots appended (approximating the manager's "state slot at
    the last computed index" shape).
    """
    shared_small_next = [1]
    per_group_next = [1] * len(group_id_limits)

    def _take_ids(group_idx: int, count: int) -> list[int]:
        if _group_factor(kv_cache_specs[group_idx]) == 1:
            counter, key = shared_small_next, 0
        else:
            counter, key = per_group_next, group_idx
        start = counter[key]
        end = start + count
        if end > group_id_limits[group_idx]:
            raise ValueError(
                f"V2 warmup overran KV group {group_idx}'s block-id space "
                f"({end - 1} > {group_id_limits[group_idx] - 1}); the "
                "warmup sizing check should have prevented this."
            )
        counter[key] = end
        return list(range(start, end))

    def _build_table(group_idx: int, num_tokens: int) -> list[int]:
        spec = kv_cache_specs[group_idx]
        rows, real = _group_table_shape(spec, num_tokens)
        if not _is_sparse_align_mamba(spec):
            return _take_ids(group_idx, rows)
        ids = _take_ids(group_idx, real)
        return [0] * (rows - 1) + [ids[0]] + ids[1:]

    def _build_delta_rows(group_idx: int, num_rows: int) -> list[int]:
        # Appended decode rows: small counts, all real distinct ids
        # (budgeted in ids_per_req; sparse groups rarely have deltas).
        return _take_ids(group_idx, num_rows) if num_rows > 0 else []

    return _build_table, _build_delta_rows


def run_mixed_prefill_decode_warmup(
    model_runner: GPUModelRunner,
    worker_execute_model: Callable[[SchedulerOutput], Any],
    worker_sample_tokens: Callable[[GrammarOutput | None], Any],
    num_tokens: int,
    *,
    mixed_step_context: AbstractContextManager[object] | None = None,
    req_id_prefix: str = "_v2_mixed_warmup",
) -> bool:
    """Run a V2 mixed prefill+decode step through normal scheduler inputs."""
    if model_runner.is_pooling_model or model_runner.max_num_reqs < 2 or num_tokens < 3:
        return False

    decode_req_id = f"{req_id_prefix}_decode_"
    prefill_req_id = f"{req_id_prefix}_prefill_"
    decode_prompt_len = 2
    decode_scheduled_tokens = 1
    prefill_len = num_tokens - decode_scheduled_tokens
    decode_token_ids = list(range(decode_prompt_len))
    prefill_token_ids = list(range(prefill_len))

    kv_cache_groups = model_runner.kv_cache_config.kv_cache_groups
    num_kv_cache_groups = len(kv_cache_groups)
    kv_cache_specs = [g.kv_cache_spec for g in kv_cache_groups]
    # Table rows and consumed real ids differ per group (sparse align-mamba
    # tables consume 1 + spec ids however many rows they have).
    decode_prefill_rows = [
        _group_table_shape(spec, decode_prompt_len)[0] for spec in kv_cache_specs
    ]
    decode_rows = [
        _group_table_shape(spec, decode_prompt_len + decode_scheduled_tokens)[0]
        for spec in kv_cache_specs
    ]
    decode_row_deltas = [
        decode - prefill for decode, prefill in zip(decode_rows, decode_prefill_rows)
    ]
    group_id_limits = _group_id_limits(model_runner, kv_cache_groups)
    # The decode request's prefill table + its appended delta rows + the
    # prefill request's table (decode-length shape would double count the
    # delta for dense groups). Small-granularity groups draw from ONE
    # shared id space, so their demand SUMS; hierarchical mamba groups
    # check their own per-group ranges.
    shared_small_required = 0
    for group_idx, (spec, limit) in enumerate(zip(kv_cache_specs, group_id_limits)):
        required_ids = (
            _group_table_shape(spec, decode_prompt_len)[1]
            + decode_row_deltas[group_idx]
            + _group_table_shape(spec, prefill_len)[1]
        )
        if _group_factor(spec) == 1:
            shared_small_required += required_ids
            required_ids = shared_small_required
        if limit <= required_ids:
            logger.warning(
                "Skipping V2 mixed prefill+decode warmup because KV group "
                "%d has only %d addressable blocks for %d required warmup "
                "blocks.",
                group_idx,
                limit,
                required_ids,
            )
            return False

    _build_table, _build_delta_rows = _make_group_table_builder(
        kv_cache_specs, group_id_limits
    )

    sampling_params = SamplingParams(max_tokens=2, temperature=0.0)

    decode_prefill_output = SchedulerOutput.make_empty()
    decode_prefill_output.scheduled_new_reqs = [
        NewRequestData(
            req_id=decode_req_id,
            prompt_token_ids=decode_token_ids,
            mm_features=[],
            sampling_params=sampling_params,
            pooling_params=None,
            block_ids=tuple(
                _build_table(g, decode_prompt_len) for g in range(num_kv_cache_groups)
            ),
            num_computed_tokens=0,
            lora_request=None,
            prefill_token_ids=decode_token_ids,
        ),
    ]
    decode_prefill_output.num_scheduled_tokens = {
        decode_req_id: decode_prompt_len,
    }
    decode_prefill_output.total_num_scheduled_tokens = decode_prompt_len
    decode_prefill_output.num_common_prefix_blocks = [0] * num_kv_cache_groups

    decode_new_blocks = tuple(
        _build_delta_rows(g, n) for g, n in enumerate(decode_row_deltas)
    )
    cached_decode_req = CachedRequestData.make_empty()
    cached_decode_req.req_ids = [decode_req_id]
    cached_decode_req.num_computed_tokens = [decode_prompt_len]
    cached_decode_req.num_output_tokens = [1]
    cached_decode_req.new_block_ids = [
        decode_new_blocks if any(decode_row_deltas) else None
    ]

    mixed_output = SchedulerOutput.make_empty()
    mixed_output.scheduled_cached_reqs = cached_decode_req
    mixed_output.scheduled_new_reqs = [
        NewRequestData(
            req_id=prefill_req_id,
            prompt_token_ids=prefill_token_ids,
            mm_features=[],
            sampling_params=sampling_params,
            pooling_params=None,
            block_ids=tuple(
                _build_table(g, prefill_len) for g in range(num_kv_cache_groups)
            ),
            num_computed_tokens=0,
            lora_request=None,
            prefill_token_ids=prefill_token_ids,
        ),
    ]
    mixed_output.num_scheduled_tokens = {
        decode_req_id: decode_scheduled_tokens,
        prefill_req_id: prefill_len,
    }
    mixed_output.total_num_scheduled_tokens = num_tokens
    mixed_output.num_common_prefix_blocks = [0] * num_kv_cache_groups

    cleanup_output = SchedulerOutput.make_empty()
    cleanup_output.finished_req_ids = {decode_req_id, prefill_req_id}

    context = mixed_step_context or nullcontext()
    model_runner.kv_connector.set_disabled(True)
    try:
        worker_execute_model(decode_prefill_output)
        worker_sample_tokens(None)
        with context:
            worker_execute_model(mixed_output)
            worker_sample_tokens(None)
        worker_execute_model(cleanup_output)
    finally:
        model_runner.kv_connector.set_disabled(False)
    return True


@torch.inference_mode()
def warmup_kernels(
    model_runner: GPUModelRunner,
    worker_execute_model: Callable[[SchedulerOutput], Any],
    worker_sample_tokens: Callable[[GrammarOutput | None], Any],
) -> None:
    """Run two execute_model + sample_tokens iterations to JIT compile
    triton kernels. We must call the provided worker's execute_model for
    pipeline parallel coordination.

    The first iteration simulates a prefill with requests of
    decode_query_len + 1 prompt tokens each. The second iteration simulates
    a decode step with all requests generating decode_query_len tokens.
    """
    num_spec_steps = model_runner.num_speculative_steps
    decode_query_len = model_runner.decode_query_len
    # Use decode_query_len + 1 tokens so the prefill batch's per-request query
    # length exceeds decode_query_len, preventing it from being misclassified as
    # a uniform decode batch.
    prompt_len = decode_query_len + 1
    prompt_token_ids = list(range(prompt_len))
    # After prefill, decode generates decode_query_len tokens.
    decode_len = prompt_len + decode_query_len

    kv_cache_groups = model_runner.kv_cache_config.kv_cache_groups
    num_kv_cache_groups = len(kv_cache_groups)

    # Encoder-decoder models: give each warmup request a dummy encoder input so
    # cross-attention warms up over a realistic, non-empty key sequence.
    # The dummy mm_feature is registered in the encoder cache and only its encoder
    # length is read (not the inputs themselves); the encoder itself is not scheduled.
    max_encoder_len = getattr(model_runner.model_state, "max_encoder_len", 0)
    warmup_mm_features: list[MultiModalFeatureSpec] = []
    if model_runner.is_encoder_decoder and max_encoder_len:
        warmup_mm_features = [
            MultiModalFeatureSpec(
                data=None,
                modality="",
                identifier="_warmup_encoder",
                mm_position=PlaceholderRange(offset=0, length=max_encoder_len),
            )
        ]

    kv_cache_specs = [g.kv_cache_spec for g in kv_cache_groups]

    def _group_tokens(num_tokens: int, spec: Any) -> int:
        # Cross-attention tables always cover the encoder length.
        return max_encoder_len if isinstance(spec, CrossAttentionSpec) else num_tokens

    prefill_rows = [
        _group_table_shape(spec, _group_tokens(prompt_len, spec))[0]
        for spec in kv_cache_specs
    ]
    decode_rows = [
        _group_table_shape(spec, _group_tokens(decode_len, spec))[0]
        for spec in kv_cache_specs
    ]
    decode_row_deltas = [d - p for d, p in zip(decode_rows, prefill_rows)]
    # Real ids one request consumes: its prefill table plus the appended
    # decode delta rows (using the decode-length shape here would double
    # count the delta for dense groups, shrinking num_reqs and skipping
    # warmup on small-but-sufficient caches).
    ids_per_req = [
        _group_table_shape(spec, _group_tokens(prompt_len, spec))[1] + delta
        for spec, delta in zip(kv_cache_specs, decode_row_deltas)
    ]
    group_id_limits = _group_id_limits(model_runner, kv_cache_groups)

    # Reserve block 0 (null block) and stay inside every id space: the
    # small-granularity groups SHARE one space (their per-request ids sum),
    # while each hierarchical mamba group has its own
    # num_blocks // large_block_factor range.
    shared_small_ids_per_req = sum(
        ids
        for spec, ids in zip(kv_cache_specs, ids_per_req)
        if _group_factor(spec) == 1
    )
    id_space_bounds = [
        (limit - 1) // ids
        for spec, limit, ids in zip(kv_cache_specs, group_id_limits, ids_per_req)
        if _group_factor(spec) > 1 and ids > 0
    ]
    if shared_small_ids_per_req > 0:
        id_space_bounds.append(
            (model_runner.kv_cache_config.num_blocks - 1) // shared_small_ids_per_req
        )
    num_reqs = min(
        model_runner.scheduler_config.max_num_seqs,
        model_runner.scheduler_config.max_num_batched_tokens
        // max(prompt_len, decode_query_len),
        min(id_space_bounds) if id_space_bounds else 1,
    )
    if num_reqs < 1:
        # Mirror run_mixed_prefill_decode_warmup: skip gracefully rather
        # than crash startup when a group cannot host even one request.
        logger.warning(
            "Skipping V2 kernel warmup because a KV group cannot host a "
            "single warmup request within its addressable block-id range."
        )
        return

    req_ids = [f"_warmup_{i}_" for i in range(num_reqs)]

    # SamplingParams exercising all sampling features.
    if model_runner.is_pooling_model:
        sampling_params = None
        pooling_params = PoolingParams()
    else:
        sampling_params = SamplingParams.for_sampler_warmup()
        pooling_params = None

    # Assign distinct block IDs per request per group. 0 null block, start
    # from 1; each group allocates within its own addressable id range.
    _build_table, _build_delta_rows = _make_group_table_builder(
        kv_cache_specs, group_id_limits
    )

    # Step 1: Prefill all requests with 1 + decode_query_len prompt tokens each.
    new_reqs = [
        NewRequestData.from_request(
            Request(
                req_ids[i],
                prompt_token_ids,
                sampling_params,
                pooling_params,
                mm_features=warmup_mm_features,
            ),
            block_ids=tuple(
                _build_table(g, _group_tokens(prompt_len, kv_cache_specs[g]))
                for g in range(num_kv_cache_groups)
            ),
            prefill_token_ids=prompt_token_ids,
        )
        for i in range(num_reqs)
    ]

    prefill_output = SchedulerOutput.make_empty()
    prefill_output.scheduled_new_reqs = new_reqs
    prefill_output.num_scheduled_tokens = {rid: prompt_len for rid in req_ids}
    prefill_output.total_num_scheduled_tokens = prompt_len * num_reqs
    prefill_output.num_common_prefix_blocks = [0] * num_kv_cache_groups

    # Disable KV connector for warmup run.
    model_runner.kv_connector.set_disabled(True)
    worker_execute_model(prefill_output)

    if not model_runner.is_pooling_model:
        # Warm up sampler and perform a decode step for non-pooling models.

        grammar_output = None
        if model_runner.is_last_pp_rank:
            # Build a GrammarOutput to exercise the structured output bitmask
            # kernel during the prefill step.
            vocab_size = model_runner.model_config.get_vocab_size()
            bitmask_width = (vocab_size + 31) // 32
            grammar_bitmask = np.full(
                (len(req_ids), bitmask_width), fill_value=-1, dtype=np.int32
            )
            grammar_output = GrammarOutput(
                structured_output_request_ids=req_ids, grammar_bitmask=grammar_bitmask
            )

        worker_sample_tokens(grammar_output)

        # Step 2: Decode all requests with decode_query_len tokens each.
        cached_req_data = CachedRequestData.make_empty()
        cached_req_data.req_ids = list(req_ids)
        cached_req_data.num_computed_tokens = [prompt_len] * num_reqs
        cached_req_data.num_output_tokens = [1] * num_reqs
        new_block = any(decode_row_deltas)
        cached_req_data.new_block_ids = [
            tuple(_build_delta_rows(g, n) for g, n in enumerate(decode_row_deltas))
            if new_block
            else None
            for _ in range(num_reqs)
        ]

        decode_output = SchedulerOutput.make_empty()
        decode_output.scheduled_cached_reqs = cached_req_data
        decode_output.num_scheduled_tokens = {
            req_id: decode_query_len for req_id in req_ids
        }
        if num_spec_steps > 0:
            decode_output.scheduled_spec_decode_tokens = {
                req_id: [0] * num_spec_steps for req_id in req_ids
            }
        decode_output.total_num_scheduled_tokens = sum(
            decode_output.num_scheduled_tokens.values()
        )
        decode_output.num_common_prefix_blocks = [0] * num_kv_cache_groups

        worker_execute_model(decode_output)
        worker_sample_tokens(None)

    # Clean up - process finish_req_ids.
    cleanup_output = SchedulerOutput.make_empty()
    cleanup_output.finished_req_ids = set(req_ids)
    worker_execute_model(cleanup_output)
    model_runner.kv_connector.set_disabled(False)
    torch.accelerator.synchronize()
