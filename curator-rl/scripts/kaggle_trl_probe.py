"""Phase B TRL spike (Roadmap v2 G.2, v3 Phase B): first contact with GRPOTrainer.

NOT a training run. It answers, with evidence written to JSON, the questions the
roadmap says must be checked against the PINNED TRL version before any adapter code is
written:

 1. versions of torch / transformers / trl / peft / datasets / accelerate / vllm;
 2. which GRPOConfig fields exist (generation, vllm, scale_rewards, loss_type, ...);
 3. what the reward function actually receives (argument names, types, whether
    completion_ids is passed, whether extra dataset columns arrive as kwargs);
 4. whether the G completions of one prompt are contiguous in the reward call
    (group integrity) and in what order prompts arrive;
 5. step time and peak GPU memory of a few real LoRA GRPO steps in fp16 on this GPU.

Every stage is wrapped so a failure is recorded with its traceback instead of
aborting; the failure text is itself the finding. Written for T4-class GPUs (fp16,
no bf16). Untested against TRL locally (TRL is not installed on the dev machine).

    python scripts/kaggle_trl_probe.py --out /kaggle/working/trl_probe
"""

from __future__ import annotations

import argparse
import dataclasses
import importlib
import inspect
import json
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
                                          "traceback": traceback.format_exc()[-3000:]}
            REPORT["stages"][name]["seconds"] = round(time.perf_counter() - t0, 2)
            status = "ok" if REPORT["stages"][name]["ok"] else "FAILED"
            print(f"[{name}] {status} ({REPORT['stages'][name]['seconds']}s)", flush=True)
        return run
    return deco


@stage("versions")
def versions():
    out = {"python": sys.version.split()[0]}
    for m in ("torch", "transformers", "trl", "peft", "datasets", "accelerate", "vllm", "bitsandbytes"):
        try:
            out[m] = getattr(importlib.import_module(m), "__version__", "?")
        except Exception as exc:  # noqa: BLE001
            out[m] = f"MISSING ({type(exc).__name__})"
    import torch

    out["cuda"] = torch.cuda.is_available()
    out["gpus"] = [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]
    out["bf16_supported"] = bool(torch.cuda.is_available() and torch.cuda.is_bf16_supported())
    return out


@stage("api_surface")
def api_surface():
    from trl import GRPOConfig, GRPOTrainer

    interesting = ("generation", "vllm", "scale_reward", "loss_type", "num_generations", "steps_per",
                   "beta", "epsilon", "mask_truncated", "max_completion", "max_prompt", "temperature",
                   "top_p", "top_k", "importance_sampling", "use_liger", "shuffle", "log_completions",
                   "reward_weights", "sync_ref", "model_init_kwargs", "remove_unused", "per_device_train")
    fields = {f.name: repr(f.default) for f in dataclasses.fields(GRPOConfig)
              if any(s in f.name for s in interesting)}
    init_params = list(inspect.signature(GRPOTrainer.__init__).parameters)
    return {
        "grpo_config_fields": fields,
        "trainer_init_params": init_params,
        "supports_rollout_func": "rollout_func" in init_params,
        "supports_callbacks": "callbacks" in init_params,
        "supports_peft_config": "peft_config" in init_params,
        "n_grpo_config_fields_total": len(dataclasses.fields(GRPOConfig)),
    }


def build_dataset(n: int, seed: int):
    from datasets import Dataset

    from curator_rl.core.config import load_config
    from curator_rl.core.seeding import SeedManager
    from curator_rl.envs.registry import EnvRegistry

    cfg = load_config(REPO_ROOT / "configs" / "base.yaml")
    reg = EnvRegistry(REPO_ROOT, cfg.data)
    rows = []
    for env_id in ("gsm8k", "countdown"):
        env = reg.get(env_id)
        for p in env.generate_batch("train", n // 2, SeedManager(seed).rng(f"probe_{env_id}")):
            rows.append({
                "prompt": [dict(m) for m in p.messages],
                "env_id": env_id,
                "prompt_id": p.prompt_id,
                "reference": p.reference,
                "meta_json": json.dumps(p.meta),
            })
    return Dataset.from_list(rows), reg


@stage("grpo_smoke")
def grpo_smoke(args):
    import torch
    from peft import LoraConfig
    from transformers import AutoModelForCausalLM, AutoTokenizer, TrainerCallback
    from trl import GRPOConfig, GRPOTrainer

    from curator_rl.core.types import Prompt

    dataset, reg = build_dataset(args.n_prompts, args.seed)
    G = args.group_size
    seen = {"calls": 0, "arg_names": None, "types": {}, "contiguous": None, "reward_seconds": [],
            "completion_ids_present": None, "batch_sizes": [], "id_sequence_head": None}

    def reward_fn(prompts, completions, **kwargs):
        t0 = time.perf_counter()
        seen["calls"] += 1
        seen["arg_names"] = sorted(["prompts", "completions", *kwargs])
        seen["types"] = {
            "prompts[0]": type(prompts[0]).__name__,
            "completions[0]": type(completions[0]).__name__,
            "completions[0][0]": type(completions[0][0]).__name__ if isinstance(completions[0], list) else None,
        }
        seen["completion_ids_present"] = "completion_ids" in kwargs
        seen["batch_sizes"].append(len(completions))
        ids = list(kwargs.get("prompt_id", []))
        if ids and seen["id_sequence_head"] is None:
            seen["id_sequence_head"] = ids[: 3 * G]
        if ids:
            # contiguity: every prompt_id forms ONE run of length G
            runs, prev = [], None
            for x in ids:
                if x != prev:
                    runs.append([x, 0])
                    prev = x
                runs[-1][1] += 1
            seen["contiguous"] = bool(len({r[0] for r in runs}) == len(runs) and all(r[1] == G for r in runs))
        rewards = []
        for i, comp in enumerate(completions):
            text = comp[0]["content"] if isinstance(comp, list) else str(comp)
            env = reg.get(kwargs["env_id"][i])
            prompt = Prompt(prompt_id=kwargs["prompt_id"][i], env_id=kwargs["env_id"][i], split="train",
                            messages=(), reference=kwargs["reference"][i], meta=json.loads(kwargs["meta_json"][i]))
            rewards.append(float(env.compute_reward(env.evaluate_response(prompt, text))))
        seen["reward_seconds"].append(round(time.perf_counter() - t0, 4))
        return rewards

    tok = AutoTokenizer.from_pretrained(args.model)
    tok.padding_side = "left"
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    try:
        model = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.float16)
    except TypeError:
        model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=torch.float16)

    cfg_kwargs = dict(
        output_dir=args.out + "/grpo_run", per_device_train_batch_size=G, num_generations=G,
        max_completion_length=args.max_completion_length, learning_rate=1e-5, max_steps=args.steps,
        fp16=True, logging_steps=1, report_to="none", save_strategy="no", remove_unused_columns=False,
        gradient_accumulation_steps=1,
    )
    valid = {f.name for f in dataclasses.fields(GRPOConfig)}
    dropped = sorted(k for k in cfg_kwargs if k not in valid)
    config = GRPOConfig(**{k: v for k, v in cfg_kwargs.items() if k in valid})

    step_times: list[float] = []

    class StepTimer(TrainerCallback):
        def on_step_begin(self, a, s, c, **k):
            self.t = time.perf_counter()
            if torch.cuda.is_available():
                torch.cuda.synchronize()

        def on_step_end(self, a, s, c, **k):
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            step_times.append(round(time.perf_counter() - self.t, 3))

    trainer = GRPOTrainer(
        model=model, reward_funcs=reward_fn, args=config, train_dataset=dataset, processing_class=tok,
        peft_config=LoraConfig(r=16, lora_alpha=32, target_modules="all-linear", task_type="CAUSAL_LM"),
        callbacks=[StepTimer()],
    )
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    trainer.train()
    return {
        "config_keys_not_in_this_trl": dropped,
        "reward_fn": {k: v for k, v in seen.items() if k != "reward_seconds"},
        "reward_fn_seconds_per_call": seen["reward_seconds"],
        "step_seconds": step_times,
        "peak_gpu_memory_gb": round(torch.cuda.max_memory_allocated() / 1e9, 2) if torch.cuda.is_available() else None,
        "log_history": [{k: v for k, v in rec.items() if isinstance(v, (int, float, str))} for rec in trainer.state.log_history][:10],
    }


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description="Phase B TRL spike")
    p.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
    p.add_argument("--n-prompts", type=int, default=16)
    p.add_argument("--group-size", type=int, default=8)
    p.add_argument("--steps", type=int, default=3)
    p.add_argument("--max-completion-length", type=int, default=256)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", default=str(REPO_ROOT / "reports" / "trl_probe"))
    args = p.parse_args(argv)
    Path(args.out).mkdir(parents=True, exist_ok=True)

    versions()
    api_surface()
    grpo_smoke(args)

    (Path(args.out) / "trl_probe.json").write_text(json.dumps(REPORT, indent=2, default=str), encoding="utf-8")
    print("\n=== summary ===")
    for name, s in REPORT["stages"].items():
        print(f"{name}: {'ok' if s['ok'] else 'FAILED - ' + s['error']}")
    smoke = REPORT["stages"].get("grpo_smoke", {})
    if smoke.get("ok"):
        r = smoke["result"]
        print(f"step seconds: {r['step_seconds']}  peak GPU GB: {r['peak_gpu_memory_gb']}")
        print(f"reward args: {r['reward_fn']['arg_names']}  completion_ids passed: {r['reward_fn']['completion_ids_present']}")
        print(f"G completions contiguous per prompt: {r['reward_fn']['contiguous']}")
    print(f"\nwrote {Path(args.out) / 'trl_probe.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
