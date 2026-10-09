"""Refit the long-horizon scenarios S-I / S-J from the Kaggle measurements (D-88, Roadmap F.7).

Every parameter that the measurements determine is derived mechanically here; parameters
they do not determine (learning speed eta, transfer) keep the values S-I already had and are
labelled as assumptions in the generated files. Inputs: `reports/pilot/phase_e_cost_probe.md`
(Qwen2.5-0.5B, vLLM, 64 dev prompts x 8, each env at its own cap) and the GRPO step time
(42.2 s per step of 16 prompts, vLLM colocate, D-83).

Derived:
- difficulty d_i from pass@1: p0 = sigmoid(slope * (0 - d)), slope 2 -> d = -logit(p0) / 2;
  Countdown (0 of 1,024 samples) keeps the zero-signal value d = 4 (p0 ~ 3e-4).
- rollout_concentration (the Beta-binomial kappa of per-prompt pass probabilities): least squares
  fit of the predicted share of mixed groups (0 < k < G) to the measured one over the three
  learnable arms.
- unit cost per prompt in GPU-hours: generation + pooled verification + policy update, where the
  update per GSM8K prompt is (42.2 s - 16 x GSM8K gen+verify) / 16 and scales with each env's
  token ratio either fully (R1) or half (R2: half fixed, half token-proportional). D-87 bracket.
- budget: ~70 rounds under Uniform (as the placeholder S-I), i.e. 70 x prompts/round x mean cost.
- benchmark weights: macro-average over domains with a test set (Roadmap v3 4.8): gsm8k, math35,
  mbpp, countdown 0.25 each; noisy 0 (train-only probe, no test set).
- nominal sizes: real train-split sizes (Static baseline), Countdown a declared 5000.

    python experiments/analysis/refit_scenarios.py

Writes configs/sim/scenario_si_r1.yaml, scenario_si_r2.yaml, scenario_sj_r.yaml and
reports/analysis/refit_scenarios.{json,md}.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
G = 8
SLOPE = 2.0
STEP_SECONDS_VLLM = 42.2          # D-83: GSM8K, 16 prompts x 8, 512 tokens
PROMPTS_PER_STEP_PROBE = 16
ROUNDS_UNDER_UNIFORM = 70
PROMPTS_PER_ROUND = 5 * 16        # simulator round shape kept from S-I (R = 5, P = 16)

# reports/pilot/phase_e_cost_probe.md, Qwen2.5-0.5B-Instruct
MEASURED = {
    #            pass@1  mixed   gen s   verify(pool) s  token ratio
    "gsm8k":     (0.350, 0.766, 0.524, 0.002, 1.00),
    "math35":    (0.168, 0.500, 1.403, 0.002, 1.88),
    "mbpp":      (0.194, 0.467, 0.317, 0.328, 0.43),
    "countdown": (0.000, 0.000, 0.535, 0.001, 0.84),
    "noisy":     (0.373, 1.000, 0.526, 0.002, 0.90),
}
NOISY_Q = 0.35                    # design value of the noisy arm (configs/base.yaml); measured 0.373 / 0.344
LEARNABLE = ("gsm8k", "math35", "mbpp")
NOMINAL = {"gsm8k": 6873, "math35": 4982, "mbpp": 314, "countdown": 5000, "noisy": 6873}
BENCH_WEIGHT = {"gsm8k": 0.25, "math35": 0.25, "mbpp": 0.25, "countdown": 0.25, "noisy": 0.0}
# NOT measured: kept from the placeholder S-I (assumptions)
ETA = {"gsm8k": 0.006, "math35": 0.005, "mbpp": 0.005, "countdown": 0.004, "noisy": 0.01}
TRANSFER = {"gsm8k": {"math35": 0.3}, "math35": {"gsm8k": 0.3}}
COUNTDOWN_DIFFICULTY = 4.0


def mixed_share(p: float, kappa: float, g: int = G) -> float:
    """P(0 < k < G) when q ~ Beta(kappa p, kappa (1-p)) and k ~ Binomial(G, q)."""
    a, b = kappa * p, kappa * (1.0 - p)
    all_right = all_wrong = 1.0
    for j in range(g):
        all_right *= (a + j) / (a + b + j)
        all_wrong *= (b + j) / (a + b + j)
    return 1.0 - all_right - all_wrong


def fit_kappa() -> tuple[float, dict[str, float]]:
    best = None
    for i in range(10, 5001):
        k = i / 100.0
        err = sum((mixed_share(MEASURED[e][0], k) - MEASURED[e][1]) ** 2 for e in LEARNABLE)
        if best is None or err < best[0]:
            best = (err, k)
    per_env = {}
    for e in LEARNABLE:
        per_env[e] = min(((mixed_share(MEASURED[e][0], i / 100.0) - MEASURED[e][1]) ** 2, i / 100.0)
                         for i in range(10, 5001))[1]
    return best[1], per_env


def difficulty(env: str) -> float:
    if env == "countdown":
        return COUNTDOWN_DIFFICULTY
    p = MEASURED[env][0]
    return -math.log(p / (1.0 - p)) / SLOPE


def unit_costs(update_fixed_share: float) -> dict[str, float]:
    """Seconds per prompt -> GPU-hours per prompt (1 GPU-hour = 1 budget unit)."""
    g_gen, g_ver = MEASURED["gsm8k"][2], MEASURED["gsm8k"][3]
    upd = (STEP_SECONDS_VLLM - PROMPTS_PER_STEP_PROBE * (g_gen + g_ver)) / PROMPTS_PER_STEP_PROBE
    out = {}
    for e, (_, _, gen, ver, tok) in MEASURED.items():
        u = upd * (update_fixed_share + (1.0 - update_fixed_share) * tok)
        out[e] = (gen + ver + u) / 3600.0
    return out


def scenario(sid: str, costs: dict[str, float], kappa: float, note: str) -> dict:
    mean_cost = sum(costs.values()) / len(costs)
    envs = []
    for e in MEASURED:
        row = {"env_id": e, "nominal_size": NOMINAL[e]}
        if e != "noisy":
            row["difficulty"] = round(difficulty(e), 4)
        row["eta"] = ETA[e]
        row["cost_usd_per_prompt"] = float(f"{costs[e]:.6g}")
        if e in TRANSFER:
            row["transfer_out"] = TRANSFER[e]
        if e == "noisy":
            row["noisy_q"] = NOISY_Q
        row["bench_weight"] = BENCH_WEIGHT[e]
        envs.append(row)
    return {
        "scenario_id": sid,
        "description": note,
        "budget_usd": float(f"{ROUNDS_UNDER_UNIFORM * PROMPTS_PER_ROUND * mean_cost:.6g}"),
        "max_rounds": 200,
        "steps_per_round": 5,
        "prompts_per_step": 16,
        "group_size": G,
        "world": {"skill_noise_std": 0.003, "cost_lognormal_sigma": 0.10,
                  "rollout_concentration": round(kappa, 2), "benchmark_items": 200},
        "calib": {"enabled": False},
        "oracle": {"simplex_step": 0.1, "random_search_draws": 1500},
        "envs": envs,
    }


HEADER = """# {sid} (D-88): refit of {base} from the Kaggle measurements (experiments/analysis/refit_scenarios.py).
# GENERATED FILE - edit the script, not this file.
# Measured (Qwen2.5-0.5B, vLLM, reports/pilot/phase_e_cost_probe.md): difficulty from pass@1,
# rollout_concentration fitted to the mixed-group shares, unit costs in GPU-hours per prompt
# ({cost_note}), benchmark = macro over domains with a test set (noisy weight 0).
# ASSUMED (not measured): eta (learning speed) and the gsm8k<->math35 transfer, kept from the
# placeholder S-I. Budget = ~70 rounds under Uniform.
"""


def main() -> int:
    kappa, per_env_kappa = fit_kappa()
    r1 = unit_costs(0.0)
    r2 = unit_costs(0.5)
    mean_r1 = sum(r1.values()) / len(r1)
    homo = {e: mean_r1 for e in r1}
    files = {
        "scenario_si_r1.yaml": ("S-I-R1", "S-I", scenario(
            "S-I-R1", r1, kappa, "S-I refit, update cost fully token-proportional (upper cost spread)"),
            "update fully token-proportional"),
        "scenario_si_r2.yaml": ("S-I-R2", "S-I", scenario(
            "S-I-R2", r2, kappa, "S-I refit, update cost half fixed (lower cost spread)"),
            "update half fixed, half token-proportional"),
        "scenario_sj_r.yaml": ("S-J-R", "S-J", scenario(
            "S-J-R", homo, kappa, "S-J refit: S-I-R1 dynamics with equal unit costs (homogeneous twin)"),
            "every arm at the S-I-R1 mean cost"),
    }
    for name, (sid, base, payload, cost_note) in files.items():
        text = HEADER.format(sid=sid, base=base, cost_note=cost_note) + yaml.safe_dump(payload, sort_keys=False)
        (REPO_ROOT / "configs" / "sim" / name).write_text(text, encoding="utf-8")

    rel = {k: {e: round(v[e] / v["gsm8k"], 3) for e in v} for k, v in (("R1", r1), ("R2", r2))}
    p0 = {e: round(1.0 / (1.0 + math.exp(SLOPE * difficulty(e))), 4) for e in MEASURED if e != "noisy"}
    fit = {e: round(mixed_share(MEASURED[e][0], kappa), 3) for e in LEARNABLE}
    report = {
        "kappa": kappa, "kappa_per_env": per_env_kappa, "kappa_placeholder": 25.0,
        "mixed_measured": {e: MEASURED[e][1] for e in LEARNABLE}, "mixed_fitted": fit,
        "mixed_at_placeholder_kappa": {e: round(mixed_share(MEASURED[e][0], 25.0), 3) for e in LEARNABLE},
        "difficulty": {e: round(difficulty(e), 4) for e in MEASURED if e != "noisy"}, "p0": p0,
        "unit_cost_gpu_hours": {"R1": r1, "R2": r2, "S-J-R": mean_r1},
        "relative_cost": rel,
        "cost_spread_learnable": {k: round(max(v[e] for e in LEARNABLE) / min(v[e] for e in LEARNABLE), 3)
                                  for k, v in (("R1", r1), ("R2", r2))},
        "budget": {sid: payload["budget_usd"] for _, (sid, _, payload, _) in files.items()},
    }
    out = REPO_ROOT / "reports" / "analysis"
    (out / "refit_scenarios.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    md = [
        "# Scenario refit from the Kaggle measurements (D-88)", "",
        "Generated by `experiments/analysis/refit_scenarios.py`.", "",
        "| env | pass@1 measured | difficulty | p0 | mixed measured | mixed fitted | mixed at kappa 25 | cost R1 (x gsm8k) | cost R2 (x gsm8k) |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for e in MEASURED:
        md.append(
            f"| {e} | {MEASURED[e][0]:.3f} | {report['difficulty'].get(e, '—')} | {p0.get(e, '—')} | "
            f"{MEASURED[e][1]:.3f} | {fit.get(e, '—')} | {report['mixed_at_placeholder_kappa'].get(e, '—')} | "
            f"{rel['R1'][e]:.2f} | {rel['R2'][e]:.2f} |")
    md += [
        "",
        f"- Fitted rollout concentration kappa = **{kappa:.2f}** (per-arm fits {per_env_kappa}); the placeholder "
        "scenarios used 25. Real prompt groups are far more polarised (all right or all wrong) than assumed, so "
        "fewer groups carry a GRPO signal and every rollout-based signal (LP, richness, mean |advantage|) is noisier.",
        f"- Cost spread among learnable arms: R1 {report['cost_spread_learnable']['R1']}x, "
        f"R2 {report['cost_spread_learnable']['R2']}x. MATH35 is the expensive arm; MBPP is the cheapest learnable arm.",
        f"- Budgets (~70 rounds under Uniform): {report['budget']}.",
        "- Assumptions kept from the placeholder S-I (not measured): eta and the gsm8k<->math35 transfer of 0.3.",
    ]
    (out / "refit_scenarios.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print("\n".join(md))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
