"""01 — smoke test: single prompt through ContinuousBatchingPipeline.generate().

Goal: confirm the pipeline loads on CPU, produces output, and populates metrics.
"""
from __future__ import annotations

import time

from common import build_text_pipeline, default_generation_config, print_metrics


def main() -> None:
    pipe = build_text_pipeline()
    cfg = default_generation_config(max_new_tokens=48)

    prompt = "In one sentence, what is continuous batching in LLM inference?"
    print(f"prompt: {prompt!r}")

    t0 = time.perf_counter()
    results = pipe.generate(prompt, cfg)
    dt = time.perf_counter() - t0

    assert len(results) == 1, f"expected 1 GenerationResult, got {len(results)}"
    r = results[0]
    texts = r.get_generation_ids()
    print(f"status: {r.m_status}")
    print(f"elapsed: {dt:.2f}s")
    print(f"output: {texts[0]!r}")
    print_metrics(pipe, "after")


if __name__ == "__main__":
    main()
