"""Phase B step-time and vLLM probe (Roadmap v3 Phase B; D-83 follow-up).

Answers, with measurements, what the planning estimate of "45-90 s per GRPO step" could
not: how long one optimizer step takes at the REAL batch shape, how that splits into
generation and update, how it scales with prompts per step, and whether vLLM colocate
works on a T4.

Stages (each recorded even if it fails):
 versions     library versions and GPU
 gen_only     HF generate for P prompts x G samples at max_new_tokens (the generation floor)
 grpo_hf      real GRPOTrainer steps with the HF backend, for each --prompts-per-step value
 grpo_vllm    the same with use_vllm=True (colocate) when vllm is importable (--backend vllm)

A GRPO optimizer step uses `per_device_train_batch_size = G` micro-batches and
`gradient_accumulation_steps = P`, so TRL generates all P x G completions in one call
(steps_per_generation defaults to the accumulation steps) and trains in micro-batches of G
(keeps logits memory small). Timing excludes the first (warm-up) step.

    python scripts/kaggle_step_probe.py --out /kaggle/working/step_probe --prompts-per-step 8,16
    python scripts/kaggle_step_probe.py --out /kaggle/working/step_probe_vllm --backend vllm
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import statistics
import sys
import time
import traceback
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

REPORT: dict = {"stages": {}}


def stage(name):
    def deco(fn):
        def run(*a, **k):
            t0 = time.perf_counter()
            try:
                REPORT["stages"][name] = {"ok": True, "result": fn(*a, **k)}
            except BaseException as exc:  # noqa: BLE001 - the failure IS the finding
                REPORT["stages"][name] = {"ok": False, "error": f"{type(exc).__name__}: {exc}",
                                          "traceback": traceback.format_exc()[-2500:]}
            REPORT["stages"][name]["seconds"] = round(time.perf_counter() - t0, 1)
            s = REPORT["stages"][name]
            print(f"[{name}] {'ok' if s['ok'] else 'FAILED'} ({s['seconds']}s)", flush=True)
        return run
    return deco


@stage("versions")
def versions():
    import importlib

    import torch

    out = {"python": sys.version.split()[0]}
    for m in ("torch", "transformers", "trl", "peft", "datasets", "accelerate", "vllm"):
        try:
            out[m] = getattr(importlib.import_module(m), "__version__", "?")
        except Exception as exc:  # noqa: BLE001
            out[m] = f"MISSING ({type(exc).__name__})"
    out["gpus"] = [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]
    return out


def load_model(name, dtype_name="float16"):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(name)
    tok.padding_side = "left"
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    dtype = getattr(torch, dtype_name)
    try:
        model = AutoModelForCausalLM.from_pretrained(name, dtype=dtype)
    except TypeError:
        model = AutoModelForCausalLM.from_pretrained(name, torch_dtype=dtype)
    return model, tok


def gsm8k_rows(n, seed):
    from curator_rl.core.config import load_config
    from curator_rl.core.seeding import SeedManager
    from curator_rl.envs.registry import EnvRegistry

    cfg = load_config(REPO_ROOT / "configs" / "base.yaml")
    env = EnvRegistry(REPO_ROOT, cfg.data).get("gsm8k")
    prompts = env.generate_batch("train", n, SeedManager(seed).rng("step_probe"))
    return env, prompts


@stage("gen_only")
def gen_only(args):
    """Generation floor: P prompts x G samples in ONE generate call (what TRL does per step)."""
    import torch

    model, tok = load_model(args.model)
    model = model.to("cuda").eval()
    _, prompts = gsm8k_rows(max(args.prompts_per_step_list), args.seed)
    out = {}
    for P in args.prompts_per_step_list:
        texts = [tok.apply_chat_template([dict(m) for m in p.messages], tokenize=False, add_generation_prompt=True)
                 for p in prompts[:P]]
        enc = tok(texts, return_tensors="pt", padding=True, add_special_tokens=False).to("cuda")
        times, toks = [], []
        for rep in range(3):
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            with torch.no_grad():
                gen = model.generate(**enc, do_sample=True, temperature=1.0, top_p=1.0, top_k=0,
                                     repetition_penalty=1.0,
                                     max_new_tokens=args.max_completion_length,
                                     num_return_sequences=args.group_size, pad_token_id=tok.pad_token_id)
            torch.cuda.synchronize()
            if rep > 0:                                   # first call is warm-up
                times.append(time.perf_counter() - t0)
                comp = gen[:, enc["input_ids"].shape[1]:]
                toks.append(int((comp != tok.pad_token_id).sum()))
        out[f"P={P}"] = {"sequences": P * args.group_size, "gen_seconds": [round(t, 2) for t in times],
                         "mean_gen_seconds": round(statistics.fmean(times), 2),
                         "completion_tokens_per_second": round(statistics.fmean(toks) / statistics.fmean(times), 1),
                         "peak_gpu_gb": round(torch.cuda.max_memory_allocated() / 1e9, 2)}
        torch.cuda.empty_cache()
    del model
    torch.cuda.empty_cache()
    return out


def run_grpo(args, P, use_vllm):
    import torch
    from datasets import Dataset
    from peft import LoraConfig
    from transformers import TrainerCallback
    from trl import GRPOConfig, GRPOTrainer

    from curator_rl.core.types import Prompt

    G = args.group_size
    env, prompts = gsm8k_rows(P * (args.steps + 1), args.seed)
    ds = Dataset.from_list([{"prompt": [dict(m) for m in p.messages], "prompt_id": p.prompt_id,
                             "reference": p.reference, "meta_json": json.dumps(p.meta)} for p in prompts])
    reward_calls = []

    def reward_fn(prompts, completions, **kw):
        t0 = time.perf_counter()
        out = []
        for i, comp in enumerate(completions):
            text = comp[0]["content"] if isinstance(comp, list) else str(comp)
            pr = Prompt(prompt_id=kw["prompt_id"][i], env_id="gsm8k", split="train", messages=(),
                        reference=kw["reference"][i], meta=json.loads(kw["meta_json"][i]))
            out.append(float(env.compute_reward(env.evaluate_response(pr, text))))
        reward_calls.append(round(time.perf_counter() - t0, 3))
        return out

    model, tok = load_model(args.model)
    cfg_kwargs = dict(
        output_dir=f"{args.out}/grpo_P{P}{'_vllm' if use_vllm else ''}", per_device_train_batch_size=G,
        gradient_accumulation_steps=P, num_generations=G, max_completion_length=args.max_completion_length,
        learning_rate=1e-5, max_steps=args.steps + 1, fp16=True, logging_steps=1, report_to="none",
        save_strategy="no", remove_unused_columns=False,
    )
    if use_vllm:
        cfg_kwargs.update(use_vllm=True, vllm_mode="colocate", vllm_gpu_memory_utilization=0.3)
    valid = {f.name for f in dataclasses.fields(GRPOConfig)}
    config = GRPOConfig(**{k: v for k, v in cfg_kwargs.items() if k in valid})
    times: list[float] = []

    class StepTimer(TrainerCallback):
        def on_step_begin(self, a, s, c, **k):
            torch.cuda.synchronize()
            self.t = time.perf_counter()

        def on_step_end(self, a, s, c, **k):
            torch.cuda.synchronize()
            times.append(round(time.perf_counter() - self.t, 2))

    trainer = GRPOTrainer(model=model, reward_funcs=reward_fn, args=config, train_dataset=ds, processing_class=tok,
                          peft_config=LoraConfig(r=16, lora_alpha=32, target_modules="all-linear", task_type="CAUSAL_LM"),
                          callbacks=[StepTimer()])
    torch.cuda.reset_peak_memory_stats()
    trainer.train()
    logs = [r for r in trainer.state.log_history if "step_time" in r]
    timed = times[1:] if len(times) > 1 else times           # drop warm-up
    mean_len = [r.get("completions/mean_length") for r in logs]
    return {"prompts_per_step": P, "completions_per_step": P * G, "step_seconds": times,
            "mean_step_seconds_after_warmup": round(statistics.fmean(timed), 2),
            "reward_fn_seconds_per_call": reward_calls,
            "mean_completion_length": [round(v, 1) for v in mean_len if v is not None],
            "clipped_ratio": [r.get("completions/clipped_ratio") for r in logs],
            "frac_reward_zero_std": [r.get("frac_reward_zero_std") for r in logs],
            "reward_mean": [r.get("reward") for r in logs],
            "peak_gpu_gb": round(torch.cuda.max_memory_allocated() / 1e9, 2)}


@stage("grpo_hf")
def grpo_hf(args):
    out = {}
    for P in args.prompts_per_step_list:
        try:
            out[f"P={P}"] = run_grpo(args, P, use_vllm=False)
        except BaseException as exc:  # noqa: BLE001 - e.g. CUDA OOM at larger P is itself a finding
            out[f"P={P}"] = {"error": f"{type(exc).__name__}: {str(exc)[:300]}"}
        import gc

        import torch

        gc.collect()
        torch.cuda.empty_cache()
    return out


@stage("grpo_vllm")
def grpo_vllm(args):
    import vllm  # noqa: F401  (raises if not installed: recorded as the finding)

    return {f"P={P}": run_grpo(args, P, use_vllm=True) for P in args.prompts_per_step_list}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Phase B step-time and vLLM probe")
    ap.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
    ap.add_argument("--prompts-per-step", default="8,16")
    ap.add_argument("--group-size", type=int, default=8)
    ap.add_argument("--max-completion-length", type=int, default=512)
    ap.add_argument("--steps", type=int, default=3, help="timed steps after one warm-up step")
    ap.add_argument("--backend", default="hf", choices=["hf", "vllm"])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=str(REPO_ROOT / "reports" / "step_probe"))
    args = ap.parse_args(argv)
    args.prompts_per_step_list = [int(x) for x in args.prompts_per_step.split(",") if x]
    Path(args.out).mkdir(parents=True, exist_ok=True)

    versions()
    if args.backend == "hf":
        gen_only(args)
        grpo_hf(args)
    else:
        grpo_vllm(args)
    REPORT["args"] = {k: v for k, v in vars(args).items()}
    (Path(args.out) / "step_probe.json").write_text(json.dumps(REPORT, indent=2, default=str), encoding="utf-8")

    print("\n=== summary ===")
    for name, s in REPORT["stages"].items():
        print(f"{name}: {'ok' if s['ok'] else 'FAILED - ' + s['error']}")
    g = REPORT["stages"].get("gen_only", {})
    if g.get("ok"):
        for k, v in g["result"].items():
            print(f"  gen only {k}: {v['mean_gen_seconds']} s for {v['sequences']} sequences "
                  f"({v['completion_tokens_per_second']} tok/s, peak {v['peak_gpu_gb']} GB)")
    for st in ("grpo_hf", "grpo_vllm"):
        s = REPORT["stages"].get(st, {})
        if s.get("ok"):
            for k, v in s["result"].items():
                if "error" in v:
                    print(f"  {st} {k}: ERROR {v['error']}")
                else:
                    print(f"  {st} {k}: {v['mean_step_seconds_after_warmup']} s/step after warm-up "
                          f"({v['completions_per_step']} completions, peak {v['peak_gpu_gb']} GB, "
                          f"clipped {v['clipped_ratio']}, zero-std {v['frac_reward_zero_std']})")
    print(f"\nwrote {Path(args.out) / 'step_probe.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
