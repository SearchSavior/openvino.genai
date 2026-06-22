"""03 — streaming outputs two ways.

  A. Callback streamer: a plain `Callable[[str], int|None]` receives each
     decoded text chunk. Return RUNNING (0 / None) to continue, STOP/CANCEL
     to abort generation early.
  B. StreamerBase subclass: override `write(token_or_tokens)` and `end()`
     for stateful streaming (token accounting, early stop by substring, etc.).

Both are passed to `ContinuousBatchingPipeline.generate()` as the `streamer`
argument. The pipeline runs streaming on a background thread so the callback
fires as tokens are produced.
"""
from __future__ import annotations

import sys
import time

import openvino_genai as ovg

from common import build_text_pipeline, default_generation_config

PROMPT = (
    "Give a concise three-bullet overview of why continuous batching helps "
    "LLM throughput. Keep each bullet under 20 words."
)


def demo_callback(pipe: ovg.ContinuousBatchingPipeline) -> None:
    print("\n--- A. callback streamer ---")
    cfg = default_generation_config(max_new_tokens=80)
    chunks: list[str] = []

    def on_chunk(text: str):
        chunks.append(text)
        sys.stdout.write(text)
        sys.stdout.flush()
        return ovg.StreamingStatus.RUNNING

    t0 = time.perf_counter()
    pipe.generate(PROMPT, cfg, on_chunk)
    print(f"\n[callback done in {time.perf_counter() - t0:.2f}s, "
          f"{len(chunks)} chunks, total_chars={sum(map(len, chunks))}]")


class CountingStreamer(ovg.StreamerBase):
    """Accumulates tokens, decodes on the fly, stops after `limit_tokens`."""

    def __init__(self, tokenizer: ovg.Tokenizer, limit_tokens: int = 40):
        super().__init__()
        self._tok = tokenizer
        self._ids: list[int] = []
        self._limit = limit_tokens

    def write(self, token_or_tokens):
        new = [int(token_or_tokens)] if isinstance(token_or_tokens, int) else list(token_or_tokens)
        self._ids.extend(new)
        text = self._tok.decode(new)
        sys.stdout.write(text)
        sys.stdout.flush()
        if len(self._ids) >= self._limit:
            return ovg.StreamingStatus.STOP
        return ovg.StreamingStatus.RUNNING

    def end(self):
        sys.stdout.write(f"\n[stream ended, {len(self._ids)} tokens buffered]\n")
        sys.stdout.flush()


def demo_streamer_class(pipe: ovg.ContinuousBatchingPipeline) -> None:
    print("\n--- B. StreamerBase subclass (early stop at 40 tokens) ---")
    cfg = default_generation_config(max_new_tokens=200)
    streamer = CountingStreamer(pipe.get_tokenizer(), limit_tokens=40)
    t0 = time.perf_counter()
    pipe.generate(PROMPT, cfg, streamer)
    print(f"[subclass done in {time.perf_counter() - t0:.2f}s]")


def main() -> None:
    pipe = build_text_pipeline()
    demo_callback(pipe)
    demo_streamer_class(pipe)


if __name__ == "__main__":
    main()
