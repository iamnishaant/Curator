# LP estimator selection study (Phase 4)

Selection rule: best Spearman rho with true skill velocity subject to
FPR <= 5% and lag <= 5 (fallback LP-A). NVS = none; P = none when constant.

## tuning phase (picked: **lp_b**)

| estimator | rho | sign acc | FPR | best lag | n |
|---|---|---|---|---|---|
| lp_a | +0.226 | +0.801 | +0.103 (4776) | 3 | 8204 |
| lp_b | +0.191 | +0.701 | +0.000 (4776) | 3 | 8204 |
| lp_c | +0.132 | +0.310 | +0.076 (4776) | 1 | 8204 |

## evaluation phase (picked: **lp_b**)

| estimator | rho | sign acc | FPR | best lag | n |
|---|---|---|---|---|---|
| lp_a | +0.266 | +0.817 | +0.104 (7045) | 3 | 12315 |
| lp_b | +0.204 | +0.718 | +0.000 (7045) | 3 | 12315 |
| lp_c | +0.158 | +0.310 | +0.077 (7045) | 1 | 12315 |
