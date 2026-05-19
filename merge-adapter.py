# %% [code]
%%writefile merge.py
from __future__ import annotations

import argparse
import json
import re
import shutil
import time
from collections import defaultdict
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch
from safetensors import safe_open
from safetensors.torch import load_file, save_file


FP4_VALUES = np.array(
    [
        0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0,
        -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0,
    ],
    dtype=np.float32,
)
FP4_BOUNDARIES = np.array([0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0], dtype=np.float32)

_numba_dequantize = None
_numba_quantize = None


def log(msg: str) -> None:
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {msg}", flush=True)


def init_numba() -> None:
    global _numba_dequantize, _numba_quantize
    if _numba_dequantize is not None:
        return

    from numba import njit, prange  # type: ignore[import-not-found]

    @njit(parallel=True, cache=True, fastmath=True)
    def dequantize(blocks, scales, fp4_values):
        n_exp, out_dim, n_scales, _ = blocks.shape
        result = np.empty((n_exp, out_dim, n_scales * 32), dtype=np.float32)
        for e in prange(n_exp):
            for o in range(out_dim):
                for s in range(n_scales):
                    scale = 2.0 ** (int(scales[e, o, s]) - 127)
                    base = s * 32
                    for p in range(16):
                        b = blocks[e, o, s, p]
                        result[e, o, base + p * 2] = fp4_values[b & 0x0F] * scale
                        result[e, o, base + p * 2 + 1] = fp4_values[b >> 4] * scale
        return result

    @njit(parallel=True, cache=True, fastmath=True)
    def quantize(tensor, n_scales, boundaries):
        n_exp, out_dim, _ = tensor.shape
        blocks = np.empty((n_exp, out_dim, n_scales, 16), dtype=np.uint8)
        scales = np.empty((n_exp, out_dim, n_scales), dtype=np.uint8)
        for e in prange(n_exp):
            for o in range(out_dim):
                for s in range(n_scales):
                    base = s * 32
                    abs_max = 0.0
                    for i in range(32):
                        v = tensor[e, o, base + i]
                        av = v if v >= 0 else -v
                        if av > abs_max:
                            abs_max = av
                    if abs_max < 1e-12:
                        exp = -127
                    else:
                        exp = int(np.ceil(np.log2(abs_max / 6.0)))
                        exp = max(-127, min(127, exp))
                    scale_val = 2.0**exp
                    scales[e, o, s] = np.uint8(exp + 127)
                    for p in range(16):
                        n0 = tensor[e, o, base + p * 2] / scale_val
                        n1 = tensor[e, o, base + p * 2 + 1] / scale_val
                        a0 = n0 if n0 >= 0 else -n0
                        a1 = n1 if n1 >= 0 else -n1
                        q0 = 0
                        q1 = 0
                        for b in range(7):
                            if a0 >= boundaries[b]:
                                q0 = b + 1
                            if a1 >= boundaries[b]:
                                q1 = b + 1
                        if n0 < 0:
                            q0 += 8
                        if n1 < 0:
                            q1 += 8
                        blocks[e, o, s, p] = np.uint8(q0 | (q1 << 4))
        return blocks, scales

    _numba_dequantize = dequantize
    _numba_quantize = quantize
    log("Numba kernels compiled")


def dequantize_mxfp4(blocks: torch.Tensor, scales: torch.Tensor) -> torch.Tensor:
    if _numba_dequantize is None:
        init_numba()
    assert _numba_dequantize is not None
    arr = _numba_dequantize(blocks.cpu().numpy(), scales.cpu().numpy(), FP4_VALUES)
    return torch.from_numpy(arr)


def quantize_mxfp4(tensor: torch.Tensor, blocks_shape: tuple[int, ...]) -> tuple[torch.Tensor, torch.Tensor]:
    if _numba_quantize is None:
        init_numba()
    assert _numba_quantize is not None
    n_scales = int(blocks_shape[2])
    blocks, scales = _numba_quantize(tensor.contiguous().cpu().numpy(), n_scales, FP4_BOUNDARIES)
    return torch.from_numpy(blocks), torch.from_numpy(scales)


def count_fp4_distribution(blocks: torch.Tensor) -> list[int]:
    b = blocks.cpu().numpy().ravel()
    counts = np.bincount(b & 0x0F, minlength=16) + np.bincount(b >> 4, minlength=16)
    return counts.tolist()


@dataclass
class NormStats:
    l2_avg: float
    l1_avg: float
    l0_avg: float


@dataclass
class MergeStats:
    precision: str
    num_elements: int
    num_changed: int
    original: NormStats
    merged: NormStats
    diff: NormStats
    sources: list[str]
    fp4_dist_before: list[int] | None = None
    fp4_dist_after: list[int] | None = None


@dataclass
class LoraInfo:
    lora_A: torch.Tensor
    lora_B: torch.Tensor
    scaling: float
    source_name: str


def norm_stats(t: torch.Tensor) -> NormStats:
    x = t.float()
    n = max(1, x.numel())
    return NormStats(
        l2_avg=float(torch.linalg.vector_norm(x).item() / (n**0.5)),
        l1_avg=float(x.abs().sum().item() / n),
        l0_avg=float((x != 0).sum().item() / n),
    )


def compute_merge_stats(
    *,
    original: torch.Tensor,
    merged: torch.Tensor,
    precision: str,
    sources: list[str],
    fp4_before: list[int] | None = None,
    fp4_after: list[int] | None = None,
) -> MergeStats:
    orig_f = original.float()
    merged_f = merged.float()
    diff = merged_f - orig_f
    n = max(1, diff.numel())
    changed = int((diff != 0).sum().item())
    return MergeStats(
        precision=precision,
        num_elements=int(n),
        num_changed=changed,
        original=norm_stats(orig_f),
        merged=norm_stats(merged_f),
        diff=norm_stats(diff),
        sources=sources,
        fp4_dist_before=fp4_before,
        fp4_dist_after=fp4_after,
    )


A_MARKERS = [".lora_A.weight", ".lora_A.default.weight"]
B_REPLACEMENTS = {
    ".lora_A.weight": ".lora_B.weight",
    ".lora_A.default.weight": ".lora_B.default.weight",
}

MOE_RE = re.compile(
    r"^model\.layers\.(?P<layer>\d+)\.mlp\.experts(?:\.(?P<expert>\d+))?\.(?P<proj>gate_proj|up_proj|down_proj)$"
)
DOWN_BLOCKS_RE = re.compile(r"^model\.layers\.(?P<layer>\d+)\.mlp\.experts\.down_proj_blocks$")
GATE_UP_BLOCKS_RE = re.compile(r"^model\.layers\.(?P<layer>\d+)\.mlp\.experts\.gate_up_proj_blocks$")


def strip_known_prefixes(name: str) -> str:
    changed = True
    while changed:
        changed = False
        for p in ("base_model.model.", "model.model."):
            if name.startswith(p):
                name = name[len(p):]
                changed = True
    name = name.replace("model.unembed_tokens", "lm_head")
    name = name.replace(".attn.", ".self_attn.")
    name = name.replace(".w1", ".gate_proj")
    name = name.replace(".w2", ".down_proj")
    name = name.replace(".w3", ".up_proj")
    return name


def split_lora_a_key(key: str) -> tuple[str, str] | None:
    for marker in A_MARKERS:
        if marker in key:
            base = key.split(marker)[0]
            b_key = key.replace(marker, B_REPLACEMENTS[marker])
            return base, b_key
    return None


def get_scaling(adapter_config: dict[str, Any]) -> float:
    r = float(adapter_config["r"])
    alpha = float(adapter_config.get("lora_alpha", r))
    return alpha / r


def build_lora_maps(
    adapter_weights: dict[str, torch.Tensor],
    adapter_config: dict[str, Any],
) -> tuple[dict[str, LoraInfo], dict[tuple[int, str], dict[str, Any]], set[str]]:
    scaling = get_scaling(adapter_config)
    bf16_map: dict[str, LoraInfo] = {}
    mxfp4_map: dict[tuple[int, str], dict[str, Any]] = defaultdict(lambda: {"shared": None, "experts": {}})
    all_sources: set[str] = set()

    for key in sorted(adapter_weights):
        pair = split_lora_a_key(key)
        if pair is None:
            continue
        raw_base, b_key = pair
        if b_key not in adapter_weights:
            raise KeyError(f"Missing LoRA B pair for {key}: expected {b_key}")

        base = strip_known_prefixes(raw_base)
        info = LoraInfo(
            lora_A=adapter_weights[key].cpu(),
            lora_B=adapter_weights[b_key].cpu(),
            scaling=scaling,
            source_name=base,
        )
        all_sources.add(base)

        m = MOE_RE.match(base)
        if m:
            layer = int(m.group("layer"))
            expert_s = m.group("expert")
            expert = None if expert_s is None else int(expert_s)
            proj = str(m.group("proj"))
            slot = mxfp4_map[(layer, proj)]
            if expert is None:
                if slot["shared"] is not None:
                    raise ValueError(f"Duplicate shared MoE LoRA for {base}")
                slot["shared"] = info
            else:
                if expert in slot["experts"]:
                    raise ValueError(f"Duplicate expert MoE LoRA for {base}")
                slot["experts"][expert] = info
            continue

        base_weight_key = base if base.endswith(".weight") else base + ".weight"
        if base_weight_key in bf16_map:
            raise ValueError(f"Duplicate BF16 LoRA target for {base_weight_key}")
        bf16_map[base_weight_key] = info

    return bf16_map, mxfp4_map, all_sources


def lora_delta(info: LoraInfo) -> torch.Tensor:
    return (info.lora_B.float() @ info.lora_A.float()) * float(info.scaling)


def merge_bf16_tensor(base: torch.Tensor, info: LoraInfo) -> torch.Tensor:
    delta = lora_delta(info)
    if tuple(delta.shape) != tuple(base.shape):
        raise ValueError(
            f"Shape mismatch for {info.source_name}: delta {tuple(delta.shape)} vs base {tuple(base.shape)}"
        )
    return (base.float() + delta).to(base.dtype)


def apply_moe_projection_delta(
    tensor3d: torch.Tensor,
    *,
    layer: int,
    proj: str,
    mxfp4_map: dict[tuple[int, str], dict[str, Any]],
) -> tuple[torch.Tensor, list[str]]:
    slot = mxfp4_map.get((layer, proj))
    if not slot:
        return tensor3d, []

    out = tensor3d.float()
    sources: list[str] = []

    shared: LoraInfo | None = slot.get("shared")
    if shared is not None:
        delta = lora_delta(shared)
        expected = tuple(out.shape[1:])
        if tuple(delta.shape) != expected:
            raise ValueError(
                f"Shared MoE shape mismatch for {shared.source_name}: delta {tuple(delta.shape)} vs expert slice {expected}"
            )
        out += delta.unsqueeze(0)
        sources.append(shared.source_name)

    experts: dict[int, LoraInfo] = slot.get("experts", {})
    n_exp = int(out.shape[0])
    for expert, info in sorted(experts.items()):
        if expert < 0 or expert >= n_exp:
            raise ValueError(f"Expert index out of range for {info.source_name}: expert={expert}, n_exp={n_exp}")
        delta = lora_delta(info)
        expected = tuple(out[expert].shape)
        if tuple(delta.shape) != expected:
            raise ValueError(
                f"Per-expert MoE shape mismatch for {info.source_name}: delta {tuple(delta.shape)} vs expert slice {expected}"
            )
        out[expert] += delta
        sources.append(info.source_name)

    return out, sources


def merge_down_proj(
    *,
    blocks: torch.Tensor,
    scales: torch.Tensor,
    layer: int,
    mxfp4_map: dict[tuple[int, str], dict[str, Any]],
) -> tuple[torch.Tensor, torch.Tensor, MergeStats | None]:
    if (layer, "down_proj") not in mxfp4_map:
        return blocks, scales, None

    fp4_before = count_fp4_distribution(blocks)
    original = dequantize_mxfp4(blocks, scales)
    merged, sources = apply_moe_projection_delta(original, layer=layer, proj="down_proj", mxfp4_map=mxfp4_map)
    merged_blocks, merged_scales = quantize_mxfp4(merged, tuple(blocks.shape))
    fp4_after = count_fp4_distribution(merged_blocks)
    merged_dq = dequantize_mxfp4(merged_blocks, merged_scales)
    stats = compute_merge_stats(
        original=original,
        merged=merged_dq,
        precision="mxfp4",
        sources=sources,
        fp4_before=fp4_before,
        fp4_after=fp4_after,
    )
    return merged_blocks.to(blocks.dtype), merged_scales.to(scales.dtype), stats


def merge_gate_up_proj(
    *,
    blocks: torch.Tensor,
    scales: torch.Tensor,
    layer: int,
    mxfp4_map: dict[tuple[int, str], dict[str, Any]],
) -> tuple[torch.Tensor, torch.Tensor, MergeStats | None]:
    has_gate = (layer, "gate_proj") in mxfp4_map
    has_up = (layer, "up_proj") in mxfp4_map
    if not (has_gate or has_up):
        return blocks, scales, None

    fp4_before = count_fp4_distribution(blocks)
    original = dequantize_mxfp4(blocks, scales)
    n_exp, fused_dim, _ = original.shape
    if fused_dim % 2 != 0:
        raise ValueError(f"gate_up_proj fused dim is not even for layer {layer}: {fused_dim}")
    half = fused_dim // 2

    merged = original.float().clone()
    all_sources: list[str] = []

    if has_gate:
        gate, sources = apply_moe_projection_delta(merged[:, :half, :], layer=layer, proj="gate_proj", mxfp4_map=mxfp4_map)
        merged[:, :half, :] = gate
        all_sources.extend(sources)

    if has_up:
        up, sources = apply_moe_projection_delta(merged[:, half:, :], layer=layer, proj="up_proj", mxfp4_map=mxfp4_map)
        merged[:, half:, :] = up
        all_sources.extend(sources)

    merged_blocks, merged_scales = quantize_mxfp4(merged, tuple(blocks.shape))
    fp4_after = count_fp4_distribution(merged_blocks)
    merged_dq = dequantize_mxfp4(merged_blocks, merged_scales)
    stats = compute_merge_stats(
        original=original,
        merged=merged_dq,
        precision="mxfp4",
        sources=all_sources,
        fp4_before=fp4_before,
        fp4_after=fp4_after,
    )
    return merged_blocks.to(blocks.dtype), merged_scales.to(scales.dtype), stats


def find_shards(base_model_dir: Path) -> list[Path]:
    shards = sorted(base_model_dir.glob("*.safetensors"))
    if not shards:
        raise FileNotFoundError(f"No .safetensors files found in {base_model_dir}")
    return shards


def copy_non_weight_files(base_model_dir: Path, out_dir: Path) -> None:
    for item in base_model_dir.iterdir():
        if item.is_file() and item.suffix != ".safetensors":
            shutil.copy2(item, out_dir / item.name)


def load_config(base_model_dir: Path) -> dict[str, Any]:
    cfg_path = base_model_dir / "config.json"
    if not cfg_path.exists():
        raise FileNotFoundError(f"config.json not found in {base_model_dir}")
    return json.loads(cfg_path.read_text(encoding="utf-8"))


def merge_model(
    *,
    base_model_dir: Path,
    adapter_dir: Path,
    out_dir: Path,
    overwrite: bool,
    allow_missing: bool,
) -> None:
    base_model_dir = base_model_dir.resolve()
    adapter_dir = adapter_dir.resolve()
    out_dir = out_dir.resolve()

    if out_dir.exists():
        if not overwrite:
            raise FileExistsError(f"Output dir already exists: {out_dir}; pass --overwrite")
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    cfg = load_config(base_model_dir)
    model_type = cfg.get("model_type")
    num_layers = cfg.get("num_hidden_layers")
    num_experts = cfg.get("num_local_experts")

    log(f"Base model dir: {base_model_dir}")
    log(f"Adapter dir:    {adapter_dir}")
    log(f"Output dir:     {out_dir}")
    log(f"Model type:     {model_type}")
    log(f"Layers:         {num_layers}")
    log(f"Local experts:  {num_experts}")

    if model_type != "gpt_oss":
        log(f"WARNING: model_type is {model_type!r}, expected 'gpt_oss'")

    adapter_config_path = adapter_dir / "adapter_config.json"
    adapter_weights_path = adapter_dir / "adapter_model.safetensors"
    if not adapter_config_path.exists():
        raise FileNotFoundError(f"Missing adapter_config.json: {adapter_config_path}")
    if not adapter_weights_path.exists():
        raise FileNotFoundError(f"Missing adapter_model.safetensors: {adapter_weights_path}")

    adapter_config = json.loads(adapter_config_path.read_text(encoding="utf-8"))
    adapter_weights = load_file(str(adapter_weights_path), device="cpu")

    log(
        f"Adapter: r={adapter_config.get('r')}, alpha={adapter_config.get('lora_alpha')}, "
        f"tensors={len(adapter_weights)}"
    )

    bf16_map, mxfp4_map, all_sources = build_lora_maps(adapter_weights, adapter_config)
    n_shared_moe = sum(1 for slot in mxfp4_map.values() if slot.get("shared") is not None)
    n_expert_moe = sum(len(slot.get("experts", {})) for slot in mxfp4_map.values())

    log("LoRA targets parsed:")
    log(f"  BF16 targets:          {len(bf16_map)}")
    log(f"  MXFP4 shared targets:  {n_shared_moe}")
    log(f"  MXFP4 expert targets:  {n_expert_moe}")
    log(f"  Total source modules:  {len(all_sources)}")

    shards = find_shards(base_model_dir)
    log(f"Found {len(shards)} base shards")

    merged_sources: set[str] = set()
    stats_by_target: dict[str, MergeStats] = {}

    copy_non_weight_files(base_model_dir, out_dir)

    for shard_idx, shard_path in enumerate(shards, start=1):
        t0 = time.time()
        log(f"Shard {shard_idx}/{len(shards)}: {shard_path.name}")

        with safe_open(str(shard_path), framework="pt", device="cpu") as f:
            tensors = {k: f.get_tensor(k) for k in f.keys()}

        out_tensors: dict[str, torch.Tensor] = dict(tensors)

        for key, info in sorted(bf16_map.items()):
            if key not in tensors:
                continue
            original = tensors[key]
            merged = merge_bf16_tensor(original, info)
            out_tensors[key] = merged
            merged_sources.add(info.source_name)
            stats_by_target[key] = compute_merge_stats(
                original=original,
                merged=merged,
                precision=str(original.dtype).replace("torch.", ""),
                sources=[info.source_name],
            )
            log(f"  BF16 merged: {key} <- {info.source_name}")

        for key in sorted(tensors):
            m = DOWN_BLOCKS_RE.match(key)
            if m:
                layer = int(m.group("layer"))
                scale_key = key.replace("_blocks", "_scales")
                if scale_key not in tensors:
                    raise KeyError(f"Missing scale tensor for {key}: {scale_key}")
                mb, ms, stats = merge_down_proj(
                    blocks=tensors[key],
                    scales=tensors[scale_key],
                    layer=layer,
                    mxfp4_map=mxfp4_map,
                )
                if stats is not None:
                    out_tensors[key] = mb
                    out_tensors[scale_key] = ms
                    merged_sources.update(stats.sources)
                    stats_by_target[f"model.layers.{layer}.mlp.experts.down_proj"] = stats
                    log(f"  MXFP4 merged: layer {layer} down_proj ({len(stats.sources)} LoRA sources)")
                continue

            m = GATE_UP_BLOCKS_RE.match(key)
            if m:
                layer = int(m.group("layer"))
                scale_key = key.replace("_blocks", "_scales")
                if scale_key not in tensors:
                    raise KeyError(f"Missing scale tensor for {key}: {scale_key}")
                mb, ms, stats = merge_gate_up_proj(
                    blocks=tensors[key],
                    scales=tensors[scale_key],
                    layer=layer,
                    mxfp4_map=mxfp4_map,
                )
                if stats is not None:
                    out_tensors[key] = mb
                    out_tensors[scale_key] = ms
                    merged_sources.update(stats.sources)
                    stats_by_target[f"model.layers.{layer}.mlp.experts.gate_up_proj"] = stats
                    log(f"  MXFP4 merged: layer {layer} gate_up_proj ({len(stats.sources)} LoRA sources)")
                continue

        out_path = out_dir / shard_path.name
        save_file(out_tensors, str(out_path), metadata={"format": "pt"})
        log(f"  Saved {out_path.name} in {time.time() - t0:.1f}s")

        del tensors
        del out_tensors

    missing = sorted(all_sources - merged_sources)
    unexpected = sorted(merged_sources - all_sources)

    log("")
    log("=== Merge completeness ===")
    log(f"Expected source modules: {len(all_sources)}")
    log(f"Merged source modules:   {len(merged_sources)}")
    log(f"Missing source modules:  {len(missing)}")
    if unexpected:
        log(f"Unexpected merged source modules: {len(unexpected)}")

    if missing:
        log("First missing modules:")
        for x in missing[:100]:
            log(f"  MISSING {x}")
        if not allow_missing:
            raise RuntimeError(
                f"{len(missing)} LoRA source modules were not merged. "
                f"Pass --allow-missing only if you intentionally want this."
            )

    stats_json = {
        "base_model_dir": str(base_model_dir),
        "adapter_dir": str(adapter_dir),
        "model_type": model_type,
        "num_hidden_layers": num_layers,
        "num_local_experts": num_experts,
        "adapter_r": adapter_config.get("r"),
        "adapter_lora_alpha": adapter_config.get("lora_alpha"),
        "n_bf16_targets": len(bf16_map),
        "n_mxfp4_shared_targets": n_shared_moe,
        "n_mxfp4_expert_targets": n_expert_moe,
        "n_expected_sources": len(all_sources),
        "n_merged_sources": len(merged_sources),
        "missing_sources": missing,
        "target_stats": {k: asdict(v) for k, v in stats_by_target.items()},
    }
    (out_dir / "merge_stats.json").write_text(json.dumps(stats_json, ensure_ascii=False, indent=2), encoding="utf-8")

    log("")
    log("=== Done ===")
    log(f"Merged model written to: {out_dir}")
    log("Use this directory as a normal HF model path in vLLM. Do NOT pass --enable-lora / --lora-modules.")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--base-model-dir", type=Path, required=True)
    p.add_argument("--adapter-dir", type=Path, required=True)
    p.add_argument("--out-dir", type=Path, required=True)
    p.add_argument("--overwrite", action="store_true")
    p.add_argument(
        "--allow-missing",
        action="store_true",
        help="Do not fail if some LoRA sources were not merged. Not recommended.",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    merge_model(
        base_model_dir=args.base_model_dir,
        adapter_dir=args.adapter_dir,
        out_dir=args.out_dir,
        overwrite=bool(args.overwrite),
        allow_missing=bool(args.allow_missing),
    )


if __name__ == "__main__":
    main()
