"""02 — parallel requests via add_request + manual step() loop.

Demonstrates:
  * Submitting N requests concurrently to the pipeline.
  * Driving the scheduler by hand with step() until every handle finishes.
  * Reading tokens incrementally from each GenerationHandle.
  * Watching batching actually happen via PipelineMetrics per step.
"""
from __future__ import annotations

import time

import openvino_genai as ovg

from common import build_text_pipeline, default_generation_config

PROMPTS = [
    "List three Linux command-line tools and one line about each:",
    "Write a haiku about paged KV cache.",
    "In two sentences, what is speculative decoding?",
    "Name two tradeoffs of int4 weight quantization.",
    "Give a one-sentence definition of continuous batching.",
]


def main() -> None:
    pipe = build_text_pipeline()
    tok = pipe.get_tokenizer()
    cfg = default_generation_config(max_new_tokens=64)

    handles: list[tuple[int, ovg.GenerationHandle]] = []
    for rid, prompt in enumerate(PROMPTS):
        handles.append((rid, pipe.add_request(rid, prompt, cfg)))
    print(f"submitted {len(handles)} requests")

    collected: dict[int, list[int]] = {rid: [] for rid, _ in handles}
    step_idx = 0
    t0 = time.perf_counter()

    while pipe.has_non_finished_requests():
        pipe.step()
        step_idx += 1
        m = pipe.get_metrics()
        active = sum(
            1 for _, h in handles if h.get_status() == ovg.GenerationStatus.RUNNING
        )
        if step_idx <= 5 or step_idx % 20 == 0:
            print(
                f"step {step_idx:3d} scheduled={m.scheduled_requests} "
                f"active={active} cache={m.cache_usage:.1f}%"
            )
        for rid, h in handles:
            if h.can_read():
                for _, out in h.read().items():
                    collected[rid].extend(out.generated_ids)

    dt = time.perf_counter() - t0
    print(f"\ntotal steps={step_idx}  wall={dt:.2f}s")
    for rid, ids in collected.items():
        text = tok.decode(ids) if ids else ""
        preview = text.replace("\n", " ")[:100]
        print(f"  [req {rid}] {len(ids):3d} toks  {preview!r}")


if __name__ == "__main__":
    main()
