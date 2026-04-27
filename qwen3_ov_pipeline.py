"""
Raw OpenVINO inference pipeline for Qwen3 (no optimum runtime abstractions).

Workflow:
1. Load the exported .xml/.bin model and inspect its inputs/outputs.
2. Build input tensors manually from a tokenized prompt.
3. Run first-token (no past KV) then subsequent tokens with KV cache.
4. Compare logits against PyTorch (CPU, seed 0) on the same prompt.

Usage:
  python3 qwen3_ov_pipeline.py --model_dir /home/user/qwen3_int8 \
                                --model_id Qwen/Qwen3-0.6B \
                                --prompt "Hello, my name is"

The script also prints the logit comparison table.
"""

import argparse
import numpy as np
import openvino as ov
from pathlib import Path
from transformers import AutoTokenizer


# ---------------------------------------------------------------------------
# Graph inspection helpers
# ---------------------------------------------------------------------------

def inspect_model(model: ov.Model) -> dict:
    """Print and return info about every input/output of the compiled model."""
    info = {"inputs": {}, "outputs": {}}
    print("\n=== MODEL INPUTS ===")
    for inp in model.inputs:
        name = inp.any_name
        shape = inp.partial_shape
        dtype = inp.element_type
        info["inputs"][name] = {"shape": shape, "dtype": dtype}
        print(f"  {name:40s}  shape={shape}  dtype={dtype}")

    print("\n=== MODEL OUTPUTS ===")
    for out in model.outputs:
        name = out.any_name
        shape = out.partial_shape
        dtype = out.element_type
        info["outputs"][name] = {"shape": shape, "dtype": dtype}
        print(f"  {name:40s}  shape={shape}  dtype={dtype}")
    print()
    return info


def find_port(model_inputs, patterns: list[str]):
    """Return the first input whose any_name matches one of the patterns (substring)."""
    for inp in model_inputs:
        n = inp.any_name.lower()
        for p in patterns:
            if p.lower() in n:
                return inp
    return None


# ---------------------------------------------------------------------------
# KV cache helpers
# ---------------------------------------------------------------------------

def build_kv_cache(model: ov.Model, batch: int, dtype=np.float32) -> dict[str, np.ndarray]:
    """
    Allocate empty KV cache tensors (past_key_values).
    The optimum export uses names like:
      past_key_values.0.key, past_key_values.0.value, ...
    with shape [batch, heads, seq, head_dim] and dynamic seq dim.
    We start with seq=0.
    """
    cache = {}
    for inp in model.inputs:
        name = inp.any_name
        if "past_key_values" not in name and "key_cache" not in name and "value_cache" not in name:
            continue
        shape = inp.partial_shape
        # Replace dynamic dims: batch stays fixed, seq starts at 0
        concrete = []
        for i, d in enumerate(shape):
            if d.is_dynamic:
                if i == 0:
                    concrete.append(batch)
                else:
                    concrete.append(0)      # empty seq for past
            else:
                concrete.append(d.get_length())
        cache[name] = np.zeros(concrete, dtype=dtype)
    return cache


# ---------------------------------------------------------------------------
# Single-step inference
# ---------------------------------------------------------------------------

def run_step(
    infer_req: ov.InferRequest,
    model: ov.Model,
    input_ids: np.ndarray,       # [batch, seq]
    attention_mask: np.ndarray,  # [batch, full_seq]
    position_ids: np.ndarray,    # [batch, seq]
    kv_cache: dict[str, np.ndarray],
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    """
    Feed inputs, run inference, return (logits, updated_kv_cache).
    """
    feeds: dict[str, np.ndarray] = {}

    # Core inputs — find by name fragment
    input_names = {inp.any_name for inp in model.inputs}

    def set_if_present(candidates, value):
        for c in candidates:
            if c in input_names:
                feeds[c] = value
                return c
        return None

    set_if_present(["input_ids"], input_ids)
    set_if_present(["attention_mask"], attention_mask)
    set_if_present(["position_ids"], position_ids)

    # KV cache
    feeds.update(kv_cache)

    # beam_idx / token_type_ids — optional
    set_if_present(["beam_idx"], np.zeros(input_ids.shape[0], dtype=np.int32))

    # Run
    infer_req.start_async(feeds)
    infer_req.wait()

    results = {out.any_name: infer_req.get_tensor(out.any_name).data.copy()
               for out in model.outputs}

    # Extract logits
    logits_name = next(n for n in results if "logits" in n.lower())
    logits = results[logits_name]

    # Update KV cache from present_key_values outputs
    new_kv: dict[str, np.ndarray] = {}
    for in_name in kv_cache:
        # optimum names: past_key_values.N.key -> present.N.key
        out_name = in_name.replace("past_key_values", "present")
        if out_name in results:
            new_kv[in_name] = results[out_name]
        else:
            # fallback: reuse old (static KV models)
            new_kv[in_name] = kv_cache[in_name]

    return logits, new_kv


# ---------------------------------------------------------------------------
# Greedy decode
# ---------------------------------------------------------------------------

def greedy_decode(
    compiled: ov.CompiledModel,
    model: ov.Model,
    input_ids_full: np.ndarray,
    max_new_tokens: int = 5,
    collect_first_logits: bool = True,
) -> tuple[list[int], np.ndarray | None]:
    """
    Run prefill + greedy decode for max_new_tokens steps.
    Returns (generated_token_ids, first_step_logits).
    """
    batch = input_ids_full.shape[0]
    infer_req = compiled.create_infer_request()
    kv_cache = build_kv_cache(model, batch)

    seq_len = input_ids_full.shape[1]
    attn_mask = np.ones((batch, seq_len), dtype=np.int64)
    pos_ids = np.arange(seq_len, dtype=np.int64)[None, :]

    # Prefill
    logits, kv_cache = run_step(infer_req, model, input_ids_full, attn_mask, pos_ids, kv_cache)
    first_logits = logits[:, -1, :].copy() if collect_first_logits else None

    generated = []
    next_token = int(np.argmax(logits[0, -1]))
    generated.append(next_token)

    past_len = seq_len
    for _ in range(max_new_tokens - 1):
        cur_ids = np.array([[next_token]], dtype=np.int64)
        cur_mask = np.ones((batch, past_len + 1), dtype=np.int64)
        cur_pos = np.array([[past_len]], dtype=np.int64)

        logits, kv_cache = run_step(infer_req, model, cur_ids, cur_mask, cur_pos, kv_cache)
        next_token = int(np.argmax(logits[0, -1]))
        generated.append(next_token)
        past_len += 1

    return generated, first_logits


# ---------------------------------------------------------------------------
# PyTorch reference
# ---------------------------------------------------------------------------

def pytorch_first_logits(model_id: str, input_ids: np.ndarray, seed: int = 0) -> np.ndarray:
    import torch
    torch.manual_seed(seed)
    from transformers import AutoModelForCausalLM
    print(f"\nLoading PyTorch reference model {model_id} on CPU...")
    pt_model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.float32)
    pt_model.eval()
    ids_t = torch.from_numpy(input_ids)
    with torch.no_grad():
        out = pt_model(ids_t)
    return out.logits[0, -1].numpy()


# ---------------------------------------------------------------------------
# Comparison
# ---------------------------------------------------------------------------

def compare_logits(pt_logits: np.ndarray, ov_logits: np.ndarray, top_k: int = 10):
    print(f"\n=== LOGIT COMPARISON (top-{top_k} by PyTorch rank) ===")
    top_idx = np.argsort(pt_logits)[::-1][:top_k]
    print(f"{'Token':>8}  {'PT logit':>12}  {'OV logit':>12}  {'diff':>10}")
    print("-" * 48)
    for i in top_idx:
        diff = pt_logits[i] - ov_logits[i]
        print(f"{i:>8d}  {pt_logits[i]:>12.4f}  {ov_logits[i]:>12.4f}  {diff:>10.4f}")

    pt_top1 = int(np.argmax(pt_logits))
    ov_top1 = int(np.argmax(ov_logits))
    print(f"\nPT top-1 token: {pt_top1}   OV top-1 token: {ov_top1}  match={pt_top1==ov_top1}")

    cos = float(
        np.dot(pt_logits, ov_logits) /
        (np.linalg.norm(pt_logits) * np.linalg.norm(ov_logits) + 1e-9)
    )
    mse = float(np.mean((pt_logits - ov_logits) ** 2))
    print(f"Cosine similarity: {cos:.6f}   MSE: {mse:.6f}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_dir", default="/home/user/qwen3_int8",
                        help="Directory with openvino_model.xml/.bin")
    parser.add_argument("--model_id", default="Qwen/Qwen3-0.6B",
                        help="HF model ID for tokenizer + PT reference")
    parser.add_argument("--prompt", default="Hello, my name is")
    parser.add_argument("--max_new_tokens", type=int, default=5)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--skip_pytorch", action="store_true",
                        help="Skip PT reference (faster, no comparison)")
    args = parser.parse_args()

    np.random.seed(args.seed)

    model_dir = Path(args.model_dir)
    xml_path = model_dir / "openvino_model.xml"
    if not xml_path.exists():
        # optimum sometimes exports with a different name
        candidates = list(model_dir.glob("*.xml"))
        if not candidates:
            raise FileNotFoundError(f"No .xml found in {model_dir}")
        xml_path = candidates[0]
    print(f"Loading OV model: {xml_path}")

    core = ov.Core()
    model = core.read_model(str(xml_path))

    # --- Inspect graph ---
    inspect_model(model)

    # --- Compile ---
    compiled = core.compile_model(model, "CPU")
    print("Model compiled on CPU.")

    # --- Tokenize ---
    tokenizer = AutoTokenizer.from_pretrained(args.model_id, trust_remote_code=True)
    enc = tokenizer(args.prompt, return_tensors="np")
    input_ids = enc["input_ids"].astype(np.int64)
    print(f"\nPrompt: {args.prompt!r}  →  {input_ids.shape[1]} tokens: {input_ids[0].tolist()}")

    # --- OV inference ---
    print("\nRunning OpenVINO greedy decode...")
    generated_ids, ov_logits = greedy_decode(
        compiled, model, input_ids, max_new_tokens=args.max_new_tokens
    )
    generated_text = tokenizer.decode(generated_ids, skip_special_tokens=True)
    print(f"Generated tokens: {generated_ids}")
    print(f"Generated text:   {generated_text!r}")

    # --- PyTorch reference ---
    if not args.skip_pytorch:
        pt_logits = pytorch_first_logits(args.model_id, input_ids, seed=args.seed)
        compare_logits(pt_logits, ov_logits[0])
    else:
        print("\n(PyTorch comparison skipped)")


if __name__ == "__main__":
    main()
