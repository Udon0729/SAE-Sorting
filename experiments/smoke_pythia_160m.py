"""End-to-end smoke test: Pythia-160m-deduped + EleutherAI SAE @ layer 5.

Purpose:
  Verify that we can (1) load Pythia-160m on cu130 sm_120 GPU, (2) run a forward
  pass on a single D_probe_v001 prompt, (3) extract residual-stream activation,
  (4) apply EleutherAI's Top-K SAE (k=32, num_latents=65536) and recover top
  feature activations, (5) report shape / stats / top-K to JSON.

This is the FIRST end-to-end exercise of the model+SAE+data stack. No
correctness claims about the activations themselves -- this only validates
plumbing.

Inputs:
  - data/probe/v001/by_category/C1.jsonl    (first row; LAMA person_attribute)
  - HF: EleutherAI/pythia-160m-deduped       (model + tokenizer)
  - HF: EleutherAI/sae-pythia-160m-deduped-32k @ layers.5/

Outputs:
  - experiments/smoke_pythia_160m_result.json
  - {torch_peak_path}        (if SCHEDULER_JOB_ID env present, writes
                              ops/queue/logs/{JOB_ID}_torch_peak.json for Phase B)

Run directly:
  CUDA_VISIBLE_DEVICES=0 uv run python experiments/smoke_pythia_160m.py

Run via scheduler:
  submit YAML with command pointing here, vram_budget_gib: 5.
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path

import safetensors.torch as st_torch
import torch
from huggingface_hub import hf_hub_download
from transformers import AutoTokenizer, GPTNeoXForCausalLM

REPO_ROOT = Path(__file__).resolve().parent.parent
PROMPT_FILE = REPO_ROOT / "data" / "probe" / "v001" / "by_category" / "C1.jsonl"
OUT_JSON = REPO_ROOT / "experiments" / "smoke_pythia_160m_result.json"

MODEL_REPO = "EleutherAI/pythia-160m-deduped"
MODEL_REVISION = "step143000"
SAE_REPO = "EleutherAI/sae-pythia-160m-deduped-32k"
LAYER = 5
DEVICE = "cuda:0"
DTYPE = torch.float32


def load_pythia():
    tok = AutoTokenizer.from_pretrained(MODEL_REPO, revision=MODEL_REVISION)
    model = GPTNeoXForCausalLM.from_pretrained(
        MODEL_REPO, revision=MODEL_REVISION, torch_dtype=DTYPE
    ).to(DEVICE).eval()
    return tok, model


def load_sae(layer: int):
    cfg_path = hf_hub_download(SAE_REPO, f"layers.{layer}/cfg.json")
    weights_path = hf_hub_download(SAE_REPO, f"layers.{layer}/sae.safetensors")
    cfg = json.loads(Path(cfg_path).read_text())
    tensors = st_torch.load_file(weights_path, device=DEVICE)
    # Cast all SAE tensors to DTYPE for consistency with model.
    tensors = {k: v.to(DTYPE) for k, v in tensors.items()}
    return cfg, tensors


def sae_topk_forward(
    x: torch.Tensor, sae_tensors: dict[str, torch.Tensor], cfg: dict
) -> dict[str, torch.Tensor]:
    """Apply EleutherAI sparsify SAE forward pass.

    Standard layout (per github.com/EleutherAI/sparsify):
      encoder.weight  (num_latents, d_in)
      encoder.bias    (num_latents,)
      W_dec           (num_latents, d_in)
      b_dec           (d_in,)             -- subtracted from input first

    Top-K (k features per token), ReLU on selected values when signed=false.
    """
    enc_w = sae_tensors["encoder.weight"]
    enc_b = sae_tensors["encoder.bias"]
    w_dec = sae_tensors["W_dec"]
    b_dec = sae_tensors["b_dec"]
    k = int(cfg["k"])
    signed = bool(cfg.get("signed", False))

    sae_in = x - b_dec
    pre_acts = sae_in @ enc_w.T + enc_b  # (B, num_latents)
    topk = pre_acts.topk(k, dim=-1, sorted=True)
    values = topk.values
    if not signed:
        values = values.relu()
    indices = topk.indices

    code = torch.zeros_like(pre_acts)
    code.scatter_(-1, indices, values)
    recon = code @ w_dec + b_dec  # (B, d_in)
    return {
        "pre_acts_summary": {
            "min": pre_acts.min().item(),
            "max": pre_acts.max().item(),
            "mean": pre_acts.mean().item(),
        },
        "topk_indices": indices,
        "topk_values": values,
        "code": code,
        "recon": recon,
    }


def write_torch_peak(job_id: str, vram_bytes_per_device: dict[int, int]) -> None:
    target = REPO_ROOT / "ops" / "queue" / "logs" / f"{job_id}_torch_peak.json"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps({str(k): int(v) for k, v in vram_bytes_per_device.items()}))


def main() -> None:
    t_start = time.time()

    # Read prompt
    with PROMPT_FILE.open() as f:
        prompt_row = json.loads(f.readline())
    prompt = prompt_row["prompt"]
    expected = prompt_row["answer"]

    # Load model
    print(f"[load] {MODEL_REPO} @ {MODEL_REVISION}")
    tok, model = load_pythia()
    n_layers = model.config.num_hidden_layers
    d_in = model.config.hidden_size
    print(f"  n_layers={n_layers}, hidden_size={d_in}")

    # Tokenize + forward
    print(f"[forward] prompt={prompt!r}")
    inputs = tok(prompt, return_tensors="pt").to(DEVICE)
    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True, use_cache=False)
    hidden_states = outputs.hidden_states  # tuple len = n_layers+1
    assert len(hidden_states) == n_layers + 1, f"unexpected hidden_states len: {len(hidden_states)}"
    # Layer L convention (sparsify): residual stream at the OUTPUT of layer L, i.e., hidden_states[L+1].
    h_layer = hidden_states[LAYER + 1]  # (1, T, d_in)
    print(f"  hidden_states[{LAYER + 1}] shape: {tuple(h_layer.shape)}")

    # Load SAE for that layer
    print(f"[load] SAE {SAE_REPO} layers.{LAYER}/")
    cfg, sae_tensors = load_sae(LAYER)
    print(f"  cfg: {cfg}")
    print(f"  sae tensor keys: {sorted(sae_tensors.keys())}")
    assert cfg["d_in"] == d_in, f"SAE d_in {cfg['d_in']} != model hidden_size {d_in}"

    # Apply SAE to last-token activation
    last_token_act = h_layer[0, -1, :]  # (d_in,)
    sae_out = sae_topk_forward(last_token_act.unsqueeze(0), sae_tensors, cfg)
    topk_indices = sae_out["topk_indices"][0].tolist()
    topk_values = sae_out["topk_values"][0].tolist()
    recon = sae_out["recon"][0]
    recon_err = (recon - last_token_act).norm().item()
    rel_err = recon_err / max(last_token_act.norm().item(), 1e-9)

    duration = time.time() - t_start

    # VRAM peak
    vram_peak = {0: torch.cuda.max_memory_allocated(0)}
    job_id = os.environ.get("SCHEDULER_JOB_ID")
    if job_id:
        write_torch_peak(job_id, vram_peak)

    result = {
        "model": {"repo": MODEL_REPO, "revision": MODEL_REVISION,
                  "n_layers": n_layers, "hidden_size": d_in},
        "sae": {"repo": SAE_REPO, "layer": LAYER, "cfg": cfg},
        "device": {"name": torch.cuda.get_device_name(0),
                   "capability": ".".join(map(str, torch.cuda.get_device_capability(0)))},
        "torch_version": torch.__version__,
        "prompt": {
            "qid": prompt_row["qid"], "category": prompt_row["category"],
            "source": prompt_row["source"], "text": prompt, "expected": expected,
        },
        "tokens": {"input_ids": inputs["input_ids"][0].tolist(),
                   "n_tokens": int(inputs["input_ids"].shape[1])},
        "activation_layer_output_index": LAYER + 1,
        "activation_stats": {
            "shape": list(h_layer.shape),
            "mean": h_layer.mean().item(), "std": h_layer.std().item(),
            "min": h_layer.min().item(), "max": h_layer.max().item(),
            "has_nan": bool(torch.isnan(h_layer).any().item()),
            "last_token_norm": last_token_act.norm().item(),
        },
        "sae_topk": {
            "k": cfg["k"],
            "indices": topk_indices,
            "values": topk_values,
            "pre_acts_summary": sae_out["pre_acts_summary"],
            "reconstruction_error_l2": recon_err,
            "reconstruction_relative_error": rel_err,
        },
        "vram_peak_gib": {str(k): round(v / (1024 ** 3), 4) for k, v in vram_peak.items()},
        "wall_time_sec": round(duration, 3),
    }

    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(result, ensure_ascii=False, indent=2))
    print(f"\n[done] wall={duration:.2f}s vram_peak={result['vram_peak_gib']} -> {OUT_JSON}")


if __name__ == "__main__":
    main()
