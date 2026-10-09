"""Per-environment unit cost under vLLM, the generation backend training will use (D-86).

The Phase B/E pilot (`kaggle_pilot.py`) timed plain HF `generate`, which runs every batch until
its LONGEST completion finishes. Short-answer environments therefore pay for the longest
sample in their batch, and the measured cost spread is compressed (MBPP averaged 136 tokens
against MATH35's 464, yet cost only 1.24x more). vLLM schedules sequences continuously, so its
wall time follows the tokens actually generated, which is what a GRPO step pays.

For every environment, on the dev split (noisy: train, D-21), with each environment's own
`max_completion_tokens` from configs/base.yaml (MATH35: 1024):

- generation seconds per prompt: `--chunks` timed `LLM.generate` calls of `--prompts-per-chunk`
  prompts x G samples each (16 x 8 = one GRPO step's generation), after one untimed warm-up;
  CV across chunks;
- verifier seconds per prompt, serial (sum of verifier times) and as the wall time of a
  `--verify-workers` thread pool (the Phase F design);
- tokens per prompt (prompt + completion; the policy update cost scales with these);
- pass@1, pass@G, mixed groups, mean |advantage|, parse rate, truncation at the env's own cap.

The sealed test split is never opened. Same prompt RNG streams as the pilot (`pilot_<env>`).

    python scripts/kaggle_cost_probe.py --model Qwen/Qwen2.5-0.5B-Instruct --out /kaggle/working/cost_probe
"""

from __future__ import annotations

import argparse
import json
import platform
import re
import statistics
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from curator_rl.core.advantage import group_mean_abs_advantage  # noqa: E402
from curator_rl.core.config import load_config  # noqa: E402
from curator_rl.core.seeding import SeedManager  # noqa: E402
from curator_rl.envs.registry import EnvRegistry  # noqa: E402

SPLIT = "dev"
SPLIT_OVERRIDE = {"noisy": "train"}   # the noisy arm is train-only by design (D-21)


def render(tok, messages) -> str:
    return tok.apply_chat_template([dict(m) for m in messages], tokenize=False, add_generation_prompt=True)


def verify_all(env, prompts, outputs, workers: int):
    """Score every completion; returns (verdicts per prompt, serial verifier seconds, pool wall seconds)."""
    jobs = [(p, c.text) for p, out in zip(prompts, outputs) for c in out.outputs]
    t0 = time.perf_counter()
    if workers > 1:
        with ThreadPoolExecutor(max_workers=workers) as pool:
            verdicts = list(pool.map(lambda j: env.evaluate_response(j[0], j[1]), jobs))
    else:
        verdicts = [env.evaluate_response(p, c) for p, c in jobs]
    wall = time.perf_counter() - t0
    serial = sum(v.verifier_seconds for v in verdicts)
    g = len(outputs[0].outputs) if outputs else 0
    grouped = [verdicts[i * g:(i + 1) * g] for i in range(len(prompts))]
    return grouped, serial, wall


def run_env(env_id, env, llm, sp_cls, tok, cap, args):
    rng = SeedManager(args.seed).rng(f"pilot_{env_id}")
    split = SPLIT_OVERRIDE.get(env_id, SPLIT)
    known = len(env.split_ids(split))
    want = (args.chunks + 1) * args.prompts_per_chunk
    n = min(want, known) if known else want
    prompts = env.generate_batch(split, n, rng)
    g = args.group_size
    # no per-request seed: vLLM's per-request generators slow sampling down; LLM(seed=...) seeds the run.
    # top_k is left at vLLM's "disabled" default, and the model's generation_config is not applied
    # to explicit SamplingParams (the HF pilot had to undo Qwen's top_k=20 / repetition_penalty, D-84)
    params = sp_cls(n=g, temperature=1.0, top_p=1.0, repetition_penalty=1.0, max_tokens=cap)

    chunks = [prompts[i:i + args.prompts_per_chunk] for i in range(0, len(prompts), args.prompts_per_chunk)]
    rows, ks, advs = [], [], []
    rewards, parsed, trunc, comp_tok, prompt_tok = [], [], [], [], []
    for ci, chunk in enumerate(chunks):
        texts = [render(tok, p.messages) for p in chunk]
        t0 = time.perf_counter()
        outs = llm.generate(texts, params, use_tqdm=False)
        gen_s = time.perf_counter() - t0
        grouped, v_serial, v_wall = verify_all(env, chunk, outs, args.verify_workers)
        for out, verdicts in zip(outs, grouped):
            k = 0
            prompt_tok.append(len(out.prompt_token_ids))
            for c, v in zip(out.outputs, verdicts):
                r = env.compute_reward(v)
                rewards.append(r >= 0.5)
                parsed.append(bool(v.parse_ok))
                comp_tok.append(len(c.token_ids))
                trunc.append(c.finish_reason == "length")
                k += int(r >= 0.5)
            ks.append(k)
            advs.append(group_mean_abs_advantage(k, g))
        rows.append({"chunk": ci, "n_prompts": len(chunk), "warmup": ci == 0, "gen_seconds": gen_s,
                     "verifier_serial_seconds": v_serial, "verifier_pool_wall_seconds": v_wall,
                     "completion_tokens": sum(len(c.token_ids) for o in outs for c in o.outputs)})

    timed = [r for r in rows if not r["warmup"]] or rows
    n_timed = sum(r["n_prompts"] for r in timed)
    gen_pp = [r["gen_seconds"] / r["n_prompts"] for r in timed]
    gen = sum(r["gen_seconds"] for r in timed) / n_timed
    ver_serial = sum(r["verifier_serial_seconds"] for r in timed) / n_timed
    ver_pool = sum(r["verifier_pool_wall_seconds"] for r in timed) / n_timed
    n_all = len(ks)
    return {
        "env_id": env_id, "split": split, "max_completion_tokens": cap, "n_prompts": n_all,
        "n_prompts_timed": n_timed, "group_size": g,
        "pass_at_1": sum(rewards) / len(rewards), "pass_at_G": sum(k >= 1 for k in ks) / n_all,
        "frac_groups_mixed": sum(0 < k < g for k in ks) / n_all, "mean_abs_advantage": sum(advs) / n_all,
        "parse_ok_rate": sum(parsed) / len(parsed), "truncation_rate": sum(trunc) / len(trunc),
        "mean_completion_tokens": sum(comp_tok) / len(comp_tok),
        "mean_prompt_tokens": sum(prompt_tok) / len(prompt_tok),
        "tokens_per_prompt": (sum(prompt_tok) + sum(comp_tok)) / n_all,  # G completions + 1 prompt each
        "gen_seconds_per_prompt": gen,
        "verifier_serial_seconds_per_prompt": ver_serial,
        "verifier_pool_seconds_per_prompt": ver_pool,
        "unit_cost_serial": gen + ver_serial,
        "unit_cost_pool": gen + ver_pool,
        "gen_cv_across_chunks": (statistics.pstdev(gen_pp) / statistics.fmean(gen_pp)) if len(gen_pp) > 1 else float("nan"),
        "k_histogram": [sum(1 for k in ks if k == v) for v in range(g + 1)],
        "chunks": rows,
    }


def markdown(results, meta) -> str:
    base = next((r for r in results if r["env_id"] == "gsm8k"), results[0])
    lines = [
        f"## Cost probe (vLLM): {meta['model']}  ({meta['gpu']}, G={meta['group_size']}, "
        f"{meta['prompts_per_chunk']} prompts/chunk x {meta['chunks']} timed chunks, verify workers={meta['verify_workers']})", "",
        "| env | cap | pass@1 | pass@G | mixed | trunc | tokens/prompt | gen s/prompt | verify s/prompt (serial / pool) "
        "| rel. cost (pool) | rel. tokens | CV |",
        "|---|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for r in results:
        lines.append(
            f"| {r['env_id']} | {r['max_completion_tokens']} | {r['pass_at_1']:.3f} | {r['pass_at_G']:.3f} | "
            f"{r['frac_groups_mixed']:.3f} | {r['truncation_rate']:.3f} | {r['tokens_per_prompt']:.0f} | "
            f"{r['gen_seconds_per_prompt']:.3f} | {r['verifier_serial_seconds_per_prompt']:.3f} / "
            f"{r['verifier_pool_seconds_per_prompt']:.3f} | {r['unit_cost_pool'] / base['unit_cost_pool']:.2f}x | "
            f"{r['tokens_per_prompt'] / base['tokens_per_prompt']:.2f}x | {r['gen_cv_across_chunks']:.2f} |"
        )
    costs = [r["unit_cost_pool"] for r in results if r["pass_at_G"] > 0]
    if costs:
        lines += ["", f"cost spread among learnable envs (max/min, pool): {max(costs) / min(costs):.2f}x"]
    return "\n".join(lines)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Per-environment unit cost under vLLM")
    ap.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
    ap.add_argument("--envs", default="gsm8k,math35,mbpp,countdown,noisy")
    ap.add_argument("--group-size", type=int, default=8)
    ap.add_argument("--prompts-per-chunk", type=int, default=16, help="one GRPO step's prompts (P)")
    ap.add_argument("--chunks", type=int, default=3, help="timed chunks per env (one more is the warm-up)")
    ap.add_argument("--max-new-tokens", type=int, default=None, help="override every env's own cap")
    ap.add_argument("--verify-workers", type=int, default=4)
    ap.add_argument("--gpu-memory-utilization", type=float, default=0.85)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=str(REPO_ROOT / "reports" / "cost_probe"))
    args = ap.parse_args(argv)

    from vllm import LLM, SamplingParams

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    cfg = load_config(REPO_ROOT / "configs" / "base.yaml")
    registry = EnvRegistry(REPO_ROOT, cfg.data)
    env_ids = [e for e in args.envs.split(",") if e]
    caps = {e: args.max_new_tokens or getattr(cfg.data.envs, e).max_completion_tokens for e in env_ids}

    llm = LLM(model=args.model, dtype="float16", seed=args.seed, gpu_memory_utilization=args.gpu_memory_utilization,
              max_model_len=max(caps.values()) + 1024)
    tok = llm.get_tokenizer()
    llm.generate(["Warm-up."], SamplingParams(max_tokens=8), use_tqdm=False)

    results = []
    for env_id in env_ids:
        print(f"-- {env_id} (cap {caps[env_id]})", flush=True)
        res = run_env(env_id, registry.get(env_id), llm, SamplingParams, tok, caps[env_id], args)
        print(f"   pass@1 {res['pass_at_1']:.3f}  gen {res['gen_seconds_per_prompt']:.3f} s/prompt  "
              f"verify {res['verifier_pool_seconds_per_prompt']:.3f} s/prompt (pool)", flush=True)
        results.append(res)

    import torch
    import vllm

    meta = {"model": args.model, "gpu": torch.cuda.get_device_name(0), "vllm": vllm.__version__,
            "torch": torch.__version__, "python": platform.python_version(), "group_size": args.group_size,
            "prompts_per_chunk": args.prompts_per_chunk, "chunks": args.chunks, "caps": caps,
            "verify_workers": args.verify_workers, "seed": args.seed,
            "utc": datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")}
    tag = re.sub(r"[^A-Za-z0-9._-]+", "_", args.model)
    (out_dir / f"cost_probe_{tag}.json").write_text(json.dumps({"meta": meta, "results": results}, indent=2),
                                                    encoding="utf-8")
    md = markdown(results, meta)
    (out_dir / f"cost_probe_{tag}.md").write_text(md + "\n", encoding="utf-8")
    print("\n" + md)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
