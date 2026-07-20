# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""End-to-end tests for align-mode mamba prefix caching on a hierarchical
(large-block) pool.

With the mamba block size decoupled from the allocation block size, a hybrid
model keeps its small attention ``block_size`` while each mamba state slot
spans ``large_block_factor`` small blocks. These tests run a live engine on a
small parallel Mamba2+attention hybrid and assert:

1. The hierarchy actually engages (``large_block_factor > 1``) and the
   attention block size is NOT inflated to the mamba state size.
2. Prefix caching ON produces token-identical greedy output to OFF, including
   resumed (cache-hitting) turns. Determinism strategy: eager + greedy +
   identical scheduling, gated by an OFF-engine run-to-run reproducibility
   check (``VLLM_BATCH_INVARIANT`` is unsupported for Mamba backends).
3. Concurrent multi-session two-turn needle recall: N sessions with distinct
   contexts prime in one batch and resume in one batch; every session must
   recall its own needle exactly, with cache hits actually occurring.
"""

import gc

import pytest

from tests.utils import create_new_process_for_each_test
from vllm import LLM, SamplingParams
from vllm.platforms import current_platform

# Reliable cache evidence: the prefix-cache prometheus
# counters lose most per-step worker metadata (upstream observability gap),
# so tests assert on engine-reported cached-token counts instead.


@pytest.fixture(autouse=True)
def _in_process_engine(monkeypatch):
    # Run the engine core in-process so tests can read the finalized config
    # (block-size resolution happens in the core, not the front-end copy).
    # Per-test (not module-level) so the setting cannot leak into other
    # modules in the same CI shard.
    monkeypatch.setenv("VLLM_ENABLE_V1_MULTIPROCESSING", "0")


MODEL = "tiiuae/Falcon-H1-0.5B-Base"
BLOCK_SIZE = 16
NUM_SESSIONS = 8

skip_unsupported = pytest.mark.skipif(
    not (current_platform.is_cuda() and current_platform.has_device_capability(80)),
    reason="Requires CUDA and >= Ampere (SM80)",
)


def _make_llm(enable_prefix_caching: bool) -> LLM:
    return LLM(
        model=MODEL,
        block_size=BLOCK_SIZE,
        # Explicit: the auto-default for models supporting mamba prefix
        # caching is "all"; this suite targets align on a hierarchical pool.
        mamba_cache_mode="align" if enable_prefix_caching else "none",
        enable_prefix_caching=enable_prefix_caching,
        enforce_eager=True,
        gpu_memory_utilization=0.35,
        max_model_len=8192,
        max_num_batched_tokens=2048,
        max_num_seqs=NUM_SESSIONS,
        disable_log_stats=False,
    )


def _shutdown(llm: LLM) -> None:
    del llm
    gc.collect()


def _filler(salt: str, approx_tokens: int) -> str:
    sentence = (
        f"The {salt} chronicle records how travelers crossed the wide river "
        f"valley, traded salt and iron, mapped the northern coast, and argued "
        f"about the meaning of the old inscriptions found near the ruined "
        f"tower above the harbor town. "
    )
    reps = approx_tokens // len(sentence.split()) + 1
    return (sentence * reps).strip()


GREEDY = SamplingParams(temperature=0.0, max_tokens=24)


@skip_unsupported
@create_new_process_for_each_test()
def test_hierarchy_engages_and_attention_block_stays_small():
    llm = _make_llm(enable_prefix_caching=True)
    try:
        cache_config = llm.llm_engine.vllm_config.cache_config
        assert cache_config.mamba_cache_mode == "align"
        # The decoupling under test: attention keeps its small block while
        # one mamba state spans many of them.
        assert cache_config.block_size == BLOCK_SIZE
        assert cache_config.mamba_large_block_factor > 1
        out = llm.generate([_filler("smoke", 128)], GREEDY)
        assert len(out[0].outputs[0].token_ids) > 0
    finally:
        _shutdown(llm)


def _top2_gap(pos_logprobs) -> float:
    """Gap in nats between the top-2 candidates at one position."""
    values = sorted((lp.logprob for lp in pos_logprobs.values()), reverse=True)
    if len(values) < 2:
        return float("inf")
    return values[0] - values[1]


def _assert_greedy_equivalent(baseline, cached, tie_margin_nats=0.05):
    """Token equality with a near-tie escape hatch.

    Exact greedy equality across compute paths is NOT a robust invariant:
    the two paths produce functionally identical logits with ~1e-2-nat
    kernel-order noise, so a statistical top-2 tie can break differently
    per path/batch shape/GPU. A mismatch is only a failure if the
    divergence position was a real decision (top-2 gap above the margin in
    BOTH engines); after a tolerated tie-flip the suffixes legitimately
    diverge, so comparison stops there."""
    assert len(baseline) == len(cached)
    for seq_idx, ((base_tokens, base_lps), (cached_tokens, cached_lps)) in enumerate(
        zip(baseline, cached)
    ):
        for pos, (base_tok, cached_tok) in enumerate(zip(base_tokens, cached_tokens)):
            if base_tok == cached_tok:
                continue
            gap = min(_top2_gap(base_lps[pos]), _top2_gap(cached_lps[pos]))
            assert gap < tie_margin_nats, (
                f"sequence {seq_idx} diverged at generated token {pos} "
                f"({base_tok} vs {cached_tok}) with a top-2 gap of "
                f"{gap:.4f} nats — a real decision, not a near-tie: "
                "prefix caching changed greedy tokens on a hierarchical pool"
            )
            break
        else:
            assert len(base_tokens) == len(cached_tokens)


@skip_unsupported
@create_new_process_for_each_test()
def test_prefix_caching_token_equivalence():
    prompts_turn1 = [_filler(f"session{i}", 2500) for i in range(4)]
    prompts_turn2 = [p + " The council then decided that " for p in prompts_turn1]
    greedy_lp = SamplingParams(temperature=0.0, max_tokens=32, logprobs=2)

    def run(llm: LLM):
        results = []
        for batch in (prompts_turn1, prompts_turn2):
            outs = llm.generate(batch, greedy_lp)
            results.extend(
                (list(o.outputs[0].token_ids), o.outputs[0].logprobs) for o in outs
            )
        return results

    llm_off = _make_llm(enable_prefix_caching=False)
    try:
        baseline = run(llm_off)
        # GATE: without reproducibility the equivalence claim is meaningless.
        # Same engine + same batch shapes -> exact equality is sound here.
        assert [t for t, _ in run(llm_off)] == [t for t, _ in baseline], (
            "OFF engine not run-to-run reproducible"
        )
    finally:
        _shutdown(llm_off)

    llm_on = _make_llm(enable_prefix_caching=True)
    try:
        _assert_greedy_equivalent(baseline, run(llm_on))
    finally:
        _shutdown(llm_on)


@skip_unsupported
@create_new_process_for_each_test()
def test_concurrent_multi_session_needle_recall():
    needles = [f"needle-{i}-{'abcdefgh'[i]}{i * 7}" for i in range(NUM_SESSIONS)]
    primes = [
        _filler(f"ctx{i}", 800)
        + f" The secret codeword is {needles[i]}. "
        + _filler(f"tail{i}", 2500)
        for i in range(NUM_SESSIONS)
    ]
    resumes = [
        p + " Repeat the secret codeword exactly: the secret codeword is"
        for p in primes
    ]

    llm = _make_llm(enable_prefix_caching=True)
    try:
        # Turn 1: prime all sessions in one batch (concurrent prefill).
        llm.generate(primes, SamplingParams(temperature=0.0, max_tokens=4))
        # Turn 2: all sessions resume concurrently; each must land on its OWN
        # cached mamba state and recall its OWN needle.
        outs = llm.generate(resumes, SamplingParams(temperature=0.0, max_tokens=24))
        failures = []
        for i, out in enumerate(outs):
            text = out.outputs[0].text
            if needles[i] not in text:
                failures.append((i, needles[i], text.strip()[:80]))
        assert not failures, f"sessions failed own-needle recall: {failures}"

        # The resumes must actually have hit the prefix cache — recall alone
        # proves nothing (a full recompute also recalls). Every turn-2
        # request must report nearly its whole prompt as cached: the
        # hierarchical align design leaves only a small-block-scale residue.
        for i, out in enumerate(outs):
            prompt_len = len(out.prompt_token_ids)
            assert out.num_cached_tokens >= 0.9 * prompt_len, (
                f"session {i}: only {out.num_cached_tokens}/{prompt_len} "
                "prompt tokens came from cache on resume"
            )
    finally:
        _shutdown(llm)
