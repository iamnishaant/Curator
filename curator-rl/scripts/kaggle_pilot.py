"""Phase B pilot: base-model pass rates and per-prompt cost per environment (Roadmap v3 Phase B).

Runs ONE model on ONE GPU (or CPU for a smoke test) with plain HF generation,
GRPO-style sampling (temperature 1.0, G samples per prompt), and measures for each
environment, on the `dev` split only (never `test`):

- pass@1 (mean observed reward), pass@G (share of prompts with >= 1 success) and the
  share of informative groups (0 < k < G) -> decides which arms are learnable and
  which are too hard (acceptance rule, Roadmap v3 4.1);
- mean |GRPO advantage| per prompt (the SEC/DUMP-style signal);
- completion length, truncation rate, parse-failure rate;
- unit cost: generation seconds per prompt (batch wall time, GPU synchronised, one
  warm-up excluded), verifier seconds per prompt, their sum, and the coefficient of
  variation across batches (Roadmap H.7 target < 10%);
- peak GPU memory, GPU name, library versions.

It does not train. Output: <out>/pilot_<model>.json, <out>/samples_<model>_<env>.jsonl
(completions for manual audit) and a markdown table on stdout.

    python scripts/kaggle_pilot.py --model Qwen/Qwen2.5-0.5B-Instruct --n-prompts 64
    python scripts/kaggle_pilot.py --model <tiny-model> --device cpu --n-prompts 4 \\
        --group-size 2 --max-new-tokens 16 --batch-prompts 2      # local smoke test
"""

from __future__ import annotations

import argparse
import json
import platform
import re
import statistics
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

import torch  # noqa: E402
from transformers import AutoModelForCausalLM, AutoTokenizer  # noqa: E402

from curator_rl.core.advantage import group_mean_abs_advantage  # noqa: E402
from curator_rl.core.config import load_config  # noqa: E402
from curator_rl.core.seeding import SeedManager  # noqa: E402
from curator_rl.envs.registry import EnvRegistry  # noqa: E402

SPLIT = "dev"          # tuning/validation split; the sealed test is never opened here
# The noisy arm is train-only by design (a wrapper over the GSM8K train partition, D-21);
# the pilot does not train, and its pass rate is fixed by construction, so `train` is safe.
SPLIT_OVERRIDE = {"noisy": "train"}
N_AUDIT_SAMPLES = 10


def render_prompt(tok, messages) -> str:
    """Chat-template the prompt; fall back to plain text for tokenizers without one (smoke tests)."""
    msgs = [dict(m) for m in messages]
    if getattr(tok, "chat_template", None):
        return tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    return "\n".join(f"{m['role']}: {m['content']}" for m in msgs) + "\nassistant:"


def sync(device: str) -> None:
    if device.startswith("cuda"):
        torch.cuda.synchronize()


def eos_id_set(model, tok) -> set[int]:
    ids = model.generation_config.eos_token_id
    if ids is None:
        ids = tok.eos_token_id
    return {int(i) for i in (ids if isinstance(ids, (list, tuple)) else [ids]) if i is not None}


def generate_group_batch(model, tok, texts, args, device, eos_ids):
    """Sample G completions for each text. Returns (completions, comp_lens, truncated, prompt_lens, seconds)."""
    enc = tok(texts, return_tensors="pt", padding=True, add_special_tokens=False).to(device)
    in_len = enc["input_ids"].shape[1]
    sync(device)
    t0 = time.perf_counter()
    with torch.no_grad():
        out = model.generate(
            **enc,
            do_sample=True,
            temperature=args.temperature,
            top_p=args.top_p,
            top_k=0,                  # TRL disables top-k; Qwen's generation_config would set 20
            repetition_penalty=1.0,   # ... and 1.05, which the first pilot silently inherited (D-84)
            max_new_tokens=args.max_new_tokens,
            num_return_sequences=args.group_size,
            pad_token_id=tok.pad_token_id,
        )
    sync(device)
    seconds = time.perf_counter() - t0
    comp = out[:, in_len:]
    lens, truncated = [], []
    for row in comp.tolist():
        first = next((i for i, t in enumerate(row) if t in eos_ids), None)
        if first is None:
            lens.append(len(row))
            truncated.append(len(row) >= args.max_new_tokens)
        else:
            lens.append(first + 1)
            truncated.append(False)
    completions = tok.batch_decode(comp, skip_special_tokens=True)
    prompt_lens = enc["attention_mask"].sum(dim=1).tolist()
    return completions, lens, truncated, prompt_lens, seconds


def run_env(env_id, env, model, tok, args, device, eos_ids, out_dir, model_tag):
    rng = SeedManager(args.seed).rng(f"pilot_{env_id}")
    split = SPLIT_OVERRIDE.get(env_id, SPLIT)
    known = len(env.split_ids(split))          # 0 for procedural environments
    n_wanted = min(args.n_prompts, known) if known else args.n_prompts
    prompts = env.generate_batch(split, n_wanted, rng)
    g = args.group_size
    batches, samples = [], []
    ks, advs, reward_flags, parse_flags, true_flags = [], [], [], [], []
    comp_lens_all, trunc_all = [], []
    prompt_tokens_total = 0
    bp = args.batch_prompts
    i = 0
    while i < len(prompts):
        chunk = prompts[i : i + bp]
        texts = [render_prompt(tok, p.messages) for p in chunk]
        try:
            completions, lens, trunc, plens, gen_s = generate_group_batch(
                model, tok, texts, args, device, eos_ids
            )
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            if bp == 1:
                raise
            bp = max(1, bp // 2)
            print(f"  [{env_id}] CUDA OOM -> batch_prompts={bp}", flush=True)
            continue
        t_v = time.perf_counter()
        v_secs = 0.0
        for j, p in enumerate(chunk):
            k = 0
            for r in range(g):
                idx = j * g + r
                verdict = env.evaluate_response(p, completions[idx])
                reward = env.compute_reward(verdict)
                v_secs += verdict.verifier_seconds
                ok = reward >= 0.5
                k += int(ok)
                reward_flags.append(ok)
                parse_flags.append(bool(verdict.parse_ok))
                true_flags.append(bool(verdict.info.get("diagnostic_true_success", verdict.success)))
                if len(samples) < N_AUDIT_SAMPLES:
                    samples.append(
                        {"prompt_id": p.prompt_id, "reward": reward, "success": verdict.success,
                         "parse_ok": verdict.parse_ok, "completion": completions[idx][:1200]}
                    )
            ks.append(k)
            advs.append(group_mean_abs_advantage(k, g) if g >= 2 else 0.0)
        verify_wall = time.perf_counter() - t_v
        comp_lens_all += lens
        trunc_all += trunc
        prompt_tokens_total += sum(plens[: len(chunk)])
        batches.append(
            {"n_prompts": len(chunk), "gen_seconds": gen_s, "verify_wall_seconds": verify_wall,
             "verifier_seconds": v_secs, "completion_tokens": sum(lens)}
        )
        i += len(chunk)

    n = len(prompts)
    gen_per_prompt = [b["gen_seconds"] / b["n_prompts"] for b in batches]
    cv = (statistics.pstdev(gen_per_prompt) / statistics.fmean(gen_per_prompt)
          if len(gen_per_prompt) > 1 and statistics.fmean(gen_per_prompt) > 0 else float("nan"))
    gen_total = sum(b["gen_seconds"] for b in batches)
    ver_total = sum(b["verifier_seconds"] for b in batches)
    hist = [sum(1 for k in ks if k == v) for v in range(g + 1)]
    (out_dir / f"samples_{model_tag}_{env_id}.jsonl").write_text(
        "\n".join(json.dumps(s) for s in samples), encoding="utf-8")
    return {
        "env_id": env_id,
        "n_prompts": n,
        "split": split,
        "group_size": g,
        "pass_at_1": sum(reward_flags) / len(reward_flags),
        "pass_at_G": sum(1 for k in ks if k >= 1) / n,
        "frac_groups_mixed": sum(1 for k in ks if 0 < k < g) / n,
        "true_success_rate": sum(true_flags) / len(true_flags),
        "mean_abs_advantage": sum(advs) / n,
        "parse_ok_rate": sum(parse_flags) / len(parse_flags),
        "mean_completion_tokens": sum(comp_lens_all) / len(comp_lens_all),
        "truncation_rate": sum(trunc_all) / len(trunc_all),
        "mean_prompt_tokens": prompt_tokens_total / n,
        "gen_seconds_per_prompt": gen_total / n,
        "verifier_seconds_per_prompt": ver_total / n,
        "unit_cost_seconds_per_prompt": (gen_total + ver_total) / n,
        "gen_cv_across_batches": cv,
        "completion_tokens_per_second": sum(comp_lens_all) / gen_total if gen_total > 0 else float("nan"),
        "k_histogram": hist,
        "n_batches": len(batches),
        "final_batch_prompts": bp,
    }


def markdown(results: list[dict], meta: dict) -> str:
    base = next((r for r in results if r["env_id"] == "gsm8k"), results[0])["unit_cost_seconds_per_prompt"]
    lines = [
        f"## Pilot: {meta['model']}  ({meta['gpu']}, {meta['dtype']}, G={meta['group_size']}, "
        f"{meta['n_prompts']} dev prompts/env, max_new_tokens={meta['max_new_tokens']})", "",
        "| env | pass@1 | pass@G | mixed groups | mean abs adv | parse ok | tokens | trunc | "
        "s/prompt (gen+verify) | rel. cost | batch CV |",
        "|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for r in results:
        lines.append(
            f"| {r['env_id']} | {r['pass_at_1']:.3f} | {r['pass_at_G']:.3f} | {r['frac_groups_mixed']:.3f} | "
            f"{r['mean_abs_advantage']:.3f} | {r['parse_ok_rate']:.3f} | {r['mean_completion_tokens']:.0f} | "
            f"{r['truncation_rate']:.3f} | {r['unit_cost_seconds_per_prompt']:.3f} | "
            f"{r['unit_cost_seconds_per_prompt'] / base:.2f}x | {r['gen_cv_across_batches']:.2f} |"
        )
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Phase B pilot: pass rates and unit cost per environment")
    p.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
    p.add_argument("--envs", default="gsm8k,countdown,noisy")
    p.add_argument("--n-prompts", type=int, default=64)
    p.add_argument("--group-size", type=int, default=8)
    p.add_argument("--batch-prompts", type=int, default=8)
    p.add_argument("--max-new-tokens", type=int, default=512)
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--top-p", type=float, default=1.0)
    p.add_argument("--dtype", default="auto", choices=["auto", "float16", "bfloat16", "float32"])
    p.add_argument("--device", default="auto")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--repo-root", default=str(REPO_ROOT))
    p.add_argument("--out", default=str(REPO_ROOT / "reports" / "pilot"))
    args = p.parse_args(argv)

    device = args.device if args.device != "auto" else ("cuda" if torch.cuda.is_available() else "cpu")
    dtype_name = args.dtype if args.dtype != "auto" else ("float16" if device.startswith("cuda") else "float32")
    dtype = getattr(torch, dtype_name)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    model_tag = re.sub(r"[^A-Za-z0-9._-]+", "_", Path(args.model).name if Path(args.model).exists() else args.model)

    print(f"loading {args.model} on {device} ({dtype_name}) ...", flush=True)
    tok = AutoTokenizer.from_pretrained(args.model)
    tok.padding_side = "left"
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    try:  # transformers >= 4.56 / 5 use `dtype`; older releases use `torch_dtype`
        model = AutoModelForCausalLM.from_pretrained(args.model, dtype=dtype)
    except TypeError:
        model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=dtype)
    model = model.to(device).eval()
    eos_ids = eos_id_set(model, tok)

    cfg = load_config(Path(args.repo_root) / "configs" / "base.yaml")
    registry = EnvRegistry(Path(args.repo_root), cfg.data)

    # warm-up (kernel compilation, allocator) so it does not pollute the first environment's timings
    warm = tok(["Warm-up."], return_tensors="pt", padding=True).to(device)
    with torch.no_grad():
        model.generate(**warm, max_new_tokens=8, do_sample=False, pad_token_id=tok.pad_token_id)
    sync(device)
    if device.startswith("cuda"):
        torch.cuda.reset_peak_memory_stats()

    results = []
    for env_id in [e for e in args.envs.split(",") if e]:
        print(f"-- {env_id}", flush=True)
        env = registry.get(env_id)
        res = run_env(env_id, env, model, tok, args, device, eos_ids, out_dir, model_tag)
        print(f"   pass@1 {res['pass_at_1']:.3f}  pass@G {res['pass_at_G']:.3f}  "
              f"{res['unit_cost_seconds_per_prompt']:.3f} s/prompt", flush=True)
        results.append(res)

    meta = {
        "model": args.model, "device": device, "dtype": dtype_name,
        "gpu": torch.cuda.get_device_name(0) if device.startswith("cuda") else "cpu",
        "peak_gpu_memory_gb": (torch.cuda.max_memory_allocated() / 1e9) if device.startswith("cuda") else None,
        "torch": torch.__version__, "python": platform.python_version(),
        "group_size": args.group_size, "n_prompts": args.n_prompts, "max_new_tokens": args.max_new_tokens,
        "temperature": args.temperature, "top_p": args.top_p, "batch_prompts": args.batch_prompts,
        "split": SPLIT, "split_override": SPLIT_OVERRIDE, "seed": args.seed, "utc": datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ"),
    }
    try:
        import transformers

        meta["transformers"] = transformers.__version__
    except Exception:  # noqa: BLE001
        pass
    report = {"meta": meta, "results": results}
    (out_dir / f"pilot_{model_tag}.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    md = markdown(results, meta)
    (out_dir / f"pilot_{model_tag}.md").write_text(md + "\n", encoding="utf-8")
    print("\n" + md)
    if meta["peak_gpu_memory_gb"] is not None:
        print(f"\npeak GPU memory: {meta['peak_gpu_memory_gb']:.2f} GB")
    print(f"\nwrote {out_dir / f'pilot_{model_tag}.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
