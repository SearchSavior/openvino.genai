"""
Data-driven OpenVINO model inspection report.
No inference — pure graph analysis via ov.Core.read_model().

Covers:
  - File metadata (sizes, paths)
  - Model I/O ports (names, shapes, dtypes)
  - Op inventory (count by type)
  - Weight / constant analysis (parameter count, precision distribution, bytes)
  - Quantization summary (int8 vs fp32 constants by layer pattern)
  - Stateful KV cache detection (Assign / ReadValue nodes)
  - MatMul / Linear layer inventory
  - Attention head estimate (via QKV weight shapes)
  - JSON export with --json

Usage:
  python3 qwen3_ov_inspect.py --model_dir /home/user/qwen3_int8
  python3 qwen3_ov_inspect.py --model_dir /home/user/qwen3_int8 --json report.json
"""

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import openvino as ov


# ---------------------------------------------------------------------------
# dtype helpers
# ---------------------------------------------------------------------------

OV_DTYPE_BYTES = {
    "f32": 4, "f16": 2, "bf16": 2,
    "i8": 1, "u8": 1, "i4": 0.5, "u4": 0.5,
    "i32": 4, "i64": 8, "boolean": 1,
}

def dtype_str(et: ov.Type) -> str:
    return et.to_string()

def dtype_bytes(et: ov.Type) -> float:
    return OV_DTYPE_BYTES.get(et.to_string(), 4)

def shape_numel(ps: ov.PartialShape) -> int | None:
    """Return element count or None if any dim is dynamic."""
    total = 1
    for d in ps:
        if d.is_dynamic:
            return None
        total *= d.get_length()
    return total


# ---------------------------------------------------------------------------
# File metadata
# ---------------------------------------------------------------------------

def file_metadata(xml_path: Path) -> dict:
    bin_path = xml_path.with_suffix(".bin")
    return {
        "xml_path": str(xml_path),
        "bin_path": str(bin_path),
        "xml_size_kb": round(xml_path.stat().st_size / 1024, 1) if xml_path.exists() else None,
        "bin_size_mb": round(bin_path.stat().st_size / 1024 / 1024, 2) if bin_path.exists() else None,
    }


# ---------------------------------------------------------------------------
# I/O port analysis
# ---------------------------------------------------------------------------

def analyze_ports(model: ov.Model) -> dict:
    inputs = []
    for port in model.inputs:
        inputs.append({
            "name": port.any_name,
            "shape": str(port.partial_shape),
            "dtype": dtype_str(port.element_type),
        })
    outputs = []
    for port in model.outputs:
        outputs.append({
            "name": port.any_name,
            "shape": str(port.partial_shape),
            "dtype": dtype_str(port.element_type),
        })
    return {"inputs": inputs, "outputs": outputs}


# ---------------------------------------------------------------------------
# Op inventory
# ---------------------------------------------------------------------------

def op_inventory(model: ov.Model) -> dict:
    counts: Counter = Counter()
    for op in model.get_ops():
        counts[op.get_type_name()] += 1
    total = sum(counts.values())
    by_type = dict(counts.most_common())
    return {"total_nodes": total, "by_type": by_type}


# ---------------------------------------------------------------------------
# Constant / weight analysis
# ---------------------------------------------------------------------------

def weight_analysis(model: ov.Model) -> dict:
    precision_bytes: defaultdict = defaultdict(int)   # dtype -> total bytes
    precision_params: defaultdict = defaultdict(int)  # dtype -> total elements
    precision_tensors: defaultdict = defaultdict(int) # dtype -> count of constant nodes

    total_params = 0
    total_bytes = 0.0
    weight_nodes = []

    for op in model.get_ops():
        if op.get_type_name() != "Constant":
            continue
        et = op.get_element_type()
        ps = op.get_output_partial_shape(0)
        numel = shape_numel(ps)
        if numel is None or numel == 0:
            continue
        # Skip scalars and tiny constants (biases, masks, etc. < 64 elements)
        if numel < 64:
            continue

        ds = dtype_str(et)
        nb = numel * dtype_bytes(et)

        precision_params[ds] += numel
        precision_bytes[ds] += nb
        precision_tensors[ds] += 1
        total_params += numel
        total_bytes += nb

        shape_list = [d.get_length() for d in ps if not d.is_dynamic]
        weight_nodes.append({
            "name": op.get_friendly_name(),
            "shape": shape_list,
            "dtype": ds,
            "params": numel,
            "bytes": nb,
        })

    # Sort largest first
    weight_nodes.sort(key=lambda x: x["params"], reverse=True)

    return {
        "total_params": total_params,
        "total_bytes": total_bytes,
        "total_mb": round(total_bytes / 1024 / 1024, 2),
        "precision_summary": {
            ds: {
                "tensors": precision_tensors[ds],
                "params": precision_params[ds],
                "mb": round(precision_bytes[ds] / 1024 / 1024, 2),
                "pct_params": round(100 * precision_params[ds] / max(total_params, 1), 1),
            }
            for ds in sorted(precision_params)
        },
        "top20_weights": weight_nodes[:20],
    }


# ---------------------------------------------------------------------------
# Quantization analysis
# ---------------------------------------------------------------------------

def quant_analysis(model: ov.Model) -> dict:
    """
    Detect quantization by looking for FakeQuantize nodes and int8/u8 constants
    that feed into Convert→MatMul patterns (typical OV int8 weight compression).
    """
    fq_count = 0
    convert_count = 0
    int8_constant_count = 0
    low_prec_dtypes = {"i8", "u8", "i4", "u4"}

    for op in model.get_ops():
        t = op.get_type_name()
        if t == "FakeQuantize":
            fq_count += 1
        elif t == "Convert":
            convert_count += 1
        elif t == "Constant":
            ds = dtype_str(op.get_element_type())
            if ds in low_prec_dtypes:
                ps = op.get_output_partial_shape(0)
                numel = shape_numel(ps)
                if numel and numel >= 64:
                    int8_constant_count += 1

    scheme = "none"
    if fq_count > 0:
        scheme = "fake_quantize (PTQ/QAT)"
    elif int8_constant_count > 0:
        scheme = "weight_compression_int8 (NNCF)"

    return {
        "scheme": scheme,
        "fake_quantize_nodes": fq_count,
        "convert_nodes": convert_count,
        "low_precision_constant_tensors": int8_constant_count,
    }


# ---------------------------------------------------------------------------
# Stateful KV cache detection
# ---------------------------------------------------------------------------

def stateful_analysis(model: ov.Model) -> dict:
    assign_nodes = []
    read_value_nodes = []
    for op in model.get_ops():
        t = op.get_type_name()
        if t == "Assign":
            assign_nodes.append(op.get_friendly_name())
        elif t == "ReadValue":
            read_value_nodes.append(op.get_friendly_name())

    is_stateful = len(assign_nodes) > 0
    # Each KV pair = 1 ReadValue + 1 Assign; num_layers = pairs / 2
    kv_pairs = min(len(assign_nodes), len(read_value_nodes))

    return {
        "is_stateful": is_stateful,
        "assign_nodes": len(assign_nodes),
        "read_value_nodes": len(read_value_nodes),
        "estimated_kv_pairs": kv_pairs,
        "estimated_layers": kv_pairs // 2 if kv_pairs >= 2 else kv_pairs,
    }


# ---------------------------------------------------------------------------
# MatMul / Linear layer inventory
# ---------------------------------------------------------------------------

def _trace_to_constant(node, max_depth: int = 6):
    """
    Walk up the input graph to find an ancestor Constant node.
    Handles chains like: u8 Const → Convert → Multiply → Reshape → Transpose → MatMul
    Returns (constant_node, shape_list) or (None, None).
    """
    visited = set()
    queue = [node]
    while queue:
        n = queue.pop(0)
        nid = id(n)
        if nid in visited:
            continue
        visited.add(nid)
        if n.get_type_name() == "Constant":
            ps = n.get_output_partial_shape(0)
            shape = [d.get_length() for d in ps if not d.is_dynamic]
            if len(shape) == len(list(ps)):  # all dims static
                return n, shape
        if len(visited) > max_depth * 4:
            break
        for i in range(n.get_input_size()):
            src = n.input(i).get_source_output().get_node()
            queue.append(src)
    return None, None


def matmul_analysis(model: ov.Model) -> dict:
    """
    Collect all MatMul nodes and infer weight shapes from their constant inputs.
    Groups by name pattern to identify Q/K/V/O projections and FFN layers.
    Traces through Convert/Multiply/Reshape chains for int8 compressed weights.
    """
    layers = []
    for op in model.get_ops():
        if op.get_type_name() not in ("MatMul", "Gemm"):
            continue
        name = op.get_friendly_name()
        in_shapes = []
        for i in range(op.get_input_size()):
            ps = op.get_input_partial_shape(i)
            in_shapes.append(str(ps))
        out_shape = str(op.get_output_partial_shape(0))

        # Find weight constant by tracing static inputs
        weight_shape = None
        for i in range(op.get_input_size()):
            src = op.input(i).get_source_output().get_node()
            _, shape = _trace_to_constant(src)
            if shape and len(shape) == 2:
                weight_shape = shape
                break

        layers.append({
            "name": name,
            "in_shapes": in_shapes,
            "out_shape": out_shape,
            "weight_shape": weight_shape,
        })

    # Classify by name pattern
    pattern_counts: Counter = Counter()
    for l in layers:
        n = l["name"].lower()
        if any(k in n for k in ("q_proj", "query")):
            pattern_counts["q_proj"] += 1
        elif any(k in n for k in ("k_proj", "key")):
            pattern_counts["k_proj"] += 1
        elif any(k in n for k in ("v_proj", "value")):
            pattern_counts["v_proj"] += 1
        elif any(k in n for k in ("o_proj", "out_proj", "dense")):
            pattern_counts["o_proj"] += 1
        elif any(k in n for k in ("gate", "up_proj")):
            pattern_counts["ffn_gate/up"] += 1
        elif "down_proj" in n:
            pattern_counts["ffn_down"] += 1
        elif any(k in n for k in ("lm_head", "embed_out")):
            pattern_counts["lm_head"] += 1
        else:
            pattern_counts["other"] += 1

    return {
        "total_matmul": len(layers),
        "by_pattern": dict(pattern_counts),
        "layers": layers,
    }


# ---------------------------------------------------------------------------
# Attention head estimate
# ---------------------------------------------------------------------------

def attention_head_estimate(model: ov.Model, matmul_info: dict, stateful_info: dict,
                            config_path: Path | None = None) -> dict:
    """
    Estimate num_heads, head_dim, hidden_size from:
      1. config.json (most reliable when present)
      2. ScaledDotProductAttention input shapes in the graph
      3. Q-projection weight shapes as fallback
    """
    result = {}

    # --- Source 1: config.json ---
    if config_path and config_path.exists():
        try:
            cfg = json.loads(config_path.read_text())
            for key in ("hidden_size", "num_hidden_layers", "num_attention_heads",
                        "num_key_value_heads", "intermediate_size", "vocab_size",
                        "head_dim", "max_position_embeddings"):
                if key in cfg:
                    result[key] = cfg[key]
        except Exception:
            pass

    # --- Source 2: ScaledDotProductAttention node shapes ---
    # Input 0 is query: [batch, num_heads, seq, head_dim]
    # Input 1 is key:   [batch, num_kv_heads, kv_seq, head_dim]
    for op in model.get_ops():
        if op.get_type_name() != "ScaledDotProductAttention":
            continue
        q_shape = op.get_input_partial_shape(0)   # [?, num_heads, ?, head_dim]
        k_shape = op.get_input_partial_shape(1)   # [?, num_kv_heads, ?, head_dim]
        dims_q = list(q_shape)
        dims_k = list(k_shape)
        if len(dims_q) >= 4 and not dims_q[1].is_dynamic and not dims_q[3].is_dynamic:
            if "num_attention_heads" not in result:
                result["num_attention_heads"] = dims_q[1].get_length()
            if "head_dim" not in result:
                result["head_dim"] = dims_q[3].get_length()
        if len(dims_k) >= 4 and not dims_k[1].is_dynamic:
            if "num_key_value_heads" not in result:
                result["num_key_value_heads"] = dims_k[1].get_length()
        break  # one SDPA node is enough

    # --- Source 3: Q-proj weight shape fallback ---
    if "hidden_size" not in result:
        for l in matmul_info["layers"]:
            n = l["name"].lower()
            if any(k in n for k in ("q_proj", "query")) and l["weight_shape"]:
                ws = l["weight_shape"]
                if len(ws) == 2:
                    result["hidden_size"] = max(ws)
                    break

    # Derive hidden_size from num_heads × head_dim if missing
    if "hidden_size" not in result:
        nh = result.get("num_attention_heads")
        hd = result.get("head_dim")
        if nh and hd:
            result["hidden_size"] = nh * hd

    # Layer count
    if "num_hidden_layers" not in result:
        result["num_hidden_layers"] = (
            stateful_info["estimated_layers"] or matmul_info["by_pattern"].get("q_proj", 0)
        )

    return result


# ---------------------------------------------------------------------------
# Report printer
# ---------------------------------------------------------------------------

def hr(char="─", width=60):
    return char * width

def print_report(report: dict):
    meta = report["file_metadata"]
    ports = report["ports"]
    ops = report["op_inventory"]
    weights = report["weights"]
    quant = report["quantization"]
    state = report["stateful"]
    mm = report["matmul"]
    attn = report["attention"]

    print(hr("═"))
    print("  OpenVINO Model Inspection Report")
    print(hr("═"))

    print(f"\n{'FILE METADATA':}")
    print(hr())
    print(f"  XML  : {meta['xml_path']}  ({meta['xml_size_kb']} KB)")
    print(f"  BIN  : {meta['bin_path']}  ({meta['bin_size_mb']} MB)")

    print(f"\nMODEL I/O")
    print(hr())
    print(f"  {'INPUT':<42} {'SHAPE':<24} DTYPE")
    for p in ports["inputs"]:
        print(f"  {p['name']:<42} {p['shape']:<24} {p['dtype']}")
    print()
    print(f"  {'OUTPUT':<42} {'SHAPE':<24} DTYPE")
    for p in ports["outputs"]:
        print(f"  {p['name']:<42} {p['shape']:<24} {p['dtype']}")

    print(f"\nOP INVENTORY  ({ops['total_nodes']} total nodes)")
    print(hr())
    for op_type, count in sorted(ops["by_type"].items(), key=lambda x: -x[1])[:25]:
        bar = "█" * min(count // max(1, ops["total_nodes"] // 40), 30)
        print(f"  {op_type:<28} {count:>5}  {bar}")
    if len(ops["by_type"]) > 25:
        print(f"  ... ({len(ops['by_type']) - 25} more types)")

    print(f"\nWEIGHT ANALYSIS")
    print(hr())
    print(f"  Total parameters  : {weights['total_params']:,}")
    print(f"  Total weight size : {weights['total_mb']} MB")
    print()
    print(f"  {'Dtype':<8} {'Tensors':>8} {'Params':>16} {'MB':>8} {'%params':>8}")
    print(f"  {'-'*8} {'-'*8} {'-'*16} {'-'*8} {'-'*8}")
    for ds, s in weights["precision_summary"].items():
        print(f"  {ds:<8} {s['tensors']:>8} {s['params']:>16,} {s['mb']:>8.1f} {s['pct_params']:>7.1f}%")

    print(f"\nQUANTIZATION")
    print(hr())
    print(f"  Scheme                    : {quant['scheme']}")
    print(f"  FakeQuantize nodes        : {quant['fake_quantize_nodes']}")
    print(f"  Convert nodes             : {quant['convert_nodes']}")
    print(f"  Low-precision constants   : {quant['low_precision_constant_tensors']}")

    print(f"\nSTATEFUL KV CACHE")
    print(hr())
    print(f"  Stateful model            : {state['is_stateful']}")
    print(f"  Assign nodes              : {state['assign_nodes']}")
    print(f"  ReadValue nodes           : {state['read_value_nodes']}")
    print(f"  Estimated KV pairs        : {state['estimated_kv_pairs']}")
    print(f"  Estimated transformer layers: {state['estimated_layers']}")

    print(f"\nMATMUL / LINEAR LAYERS  ({mm['total_matmul']} total)")
    print(hr())
    for pattern, count in sorted(mm["by_pattern"].items(), key=lambda x: -x[1]):
        print(f"  {pattern:<20} {count:>4}")

    print(f"\nARCHITECTURE")
    print(hr())
    key_order = [
        "hidden_size", "num_hidden_layers", "num_attention_heads",
        "num_key_value_heads", "head_dim", "intermediate_size",
        "vocab_size", "max_position_embeddings",
    ]
    printed = set()
    for k in key_order:
        if k in attn:
            label = k.replace("_", " ").title()
            print(f"  {label:<28} : {attn[k]}")
            printed.add(k)
    for k, v in attn.items():
        if k not in printed:
            label = k.replace("_", " ").title()
            print(f"  {label:<28} : {v}")

    print(f"\n{hr('═')}")
    print()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_dir", default="/home/user/qwen3_int8",
                        help="Directory containing openvino_model.xml/.bin")
    parser.add_argument("--xml", default=None,
                        help="Direct path to .xml (overrides --model_dir)")
    parser.add_argument("--json", default=None, metavar="PATH",
                        help="Also write full report as JSON to this path")
    parser.add_argument("--no_matmul_detail", action="store_true",
                        help="Omit per-layer MatMul list from JSON (reduces size)")
    args = parser.parse_args()

    if args.xml:
        xml_path = Path(args.xml)
    else:
        model_dir = Path(args.model_dir)
        xml_path = model_dir / "openvino_model.xml"
        if not xml_path.exists():
            candidates = list(model_dir.glob("*.xml"))
            if not candidates:
                print(f"ERROR: no .xml found in {model_dir}", file=sys.stderr)
                sys.exit(1)
            xml_path = sorted(candidates)[0]

    print(f"Reading model: {xml_path}")
    core = ov.Core()
    model = core.read_model(str(xml_path))
    print("Graph loaded — running analysis...\n")

    config_path = xml_path.parent / "config.json"

    meta  = file_metadata(xml_path)
    ports = analyze_ports(model)
    ops   = op_inventory(model)
    wts   = weight_analysis(model)
    quant = quant_analysis(model)
    state = stateful_analysis(model)
    mm    = matmul_analysis(model)
    attn  = attention_head_estimate(model, mm, state, config_path)

    report = {
        "file_metadata": meta,
        "ports": ports,
        "op_inventory": ops,
        "weights": wts,
        "quantization": quant,
        "stateful": state,
        "matmul": mm,
        "attention": attn,
    }

    print_report(report)

    if args.json:
        out = report.copy()
        if args.no_matmul_detail:
            out["matmul"] = {k: v for k, v in mm.items() if k != "layers"}
        # weight top20 already trimmed
        with open(args.json, "w") as f:
            json.dump(out, f, indent=2, default=str)
        print(f"JSON report written to {args.json}")


if __name__ == "__main__":
    main()
