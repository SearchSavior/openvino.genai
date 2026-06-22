"""Shared helpers for the continuous-batching demo scripts."""
from __future__ import annotations

import os
from pathlib import Path

import openvino_genai as ovg

TEXT_MODEL_ID = "Echo9Zulu/Nanbeige4.1-3B-int4-awq-ov"
VLM_MODEL_ID = "Echo9Zulu/Qwen3-VL-4B-Instruct-int4_asym-ov"

MODELS_ROOT = Path(os.environ.get("OVG_MODELS_ROOT", "/home/user/models"))


def model_path(repo_id: str) -> Path:
    local = MODELS_ROOT / repo_id.split("/", 1)[1]
    if not (local / "openvino_model.xml").exists():
        from huggingface_hub import snapshot_download
        snapshot_download(repo_id=repo_id, local_dir=str(local))
    return local


def default_scheduler_config(**overrides) -> ovg.SchedulerConfig:
    cfg = ovg.SchedulerConfig()
    cfg.cache_size = 1
    cfg.max_num_batched_tokens = 256
    cfg.max_num_seqs = 16
    cfg.dynamic_split_fuse = True
    for k, v in overrides.items():
        setattr(cfg, k, v)
    return cfg


def build_text_pipeline(scheduler_config=None, device="CPU", properties=None):
    return ovg.ContinuousBatchingPipeline(
        str(model_path(TEXT_MODEL_ID)),
        scheduler_config or default_scheduler_config(),
        device,
        properties or {},
    )


def print_metrics(pipe: ovg.ContinuousBatchingPipeline, label: str = "") -> None:
    m = pipe.get_metrics()
    tag = f"[{label}] " if label else ""
    print(
        f"{tag}requests={m.requests} scheduled={m.scheduled_requests} "
        f"cache={m.cache_usage:.1f}% avg={m.avg_cache_usage:.1f}% max={m.max_cache_usage:.1f}%"
    )


def default_generation_config(max_new_tokens: int = 64, **overrides) -> ovg.GenerationConfig:
    cfg = ovg.GenerationConfig()
    cfg.max_new_tokens = max_new_tokens
    cfg.do_sample = False
    for k, v in overrides.items():
        setattr(cfg, k, v)
    return cfg
