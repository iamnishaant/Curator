# Gate 2 verification report

Generated 20261009T092523Z; seeds 100..149;
config: {"gamma": 0.95, "exploration_coef": 0.5, "tau": 0.3, "epsilon": 0.1}

## Final scores (mean over seeds)

| scenario | uniform | static | lp | ucb | curator |
|---|---|---|---|---|---|
| S-A | 0.5786 | 0.6081 | 0.6096 | 0.6238 | 0.6404 |
| S-B | 0.4052 | 0.4178 | 0.4174 | 0.4210 | 0.4162 |
| S-C | 0.3442 | 0.3470 | 0.3409 | 0.3573 | 0.3985 |

## Criteria

- **a_sa_improvement**: pass=True — `{"mean_diff": 0.061833333333333323, "ci95": [0.050599999999999985, 0.07296666666666665]}`
- **b_sb_no_underperformance**: pass=True — `{"mean_diff": 0.010974999999999992, "ci95": [0.006924999999999995, 0.014900000000000002], "delta_gap": 0.01}`
- **c_junk_share**: pass=True — `{"junk_env": "too_hard", "floor": 0.0125, "threshold": 0.018750000000000003, "mean_share": 0.0125, "seeds_ok": 50, "n_seeds": 50}`
- **d_status_quality**: pass=True — `{"macro_f1": 0.9324501027408183, "pooled_flip": 0.0}`
- **e_compute_to_target**: pass=True — `{"target": 0.344375, "mean_improvement": 0.26228215953169765, "ci95": [0.22174363288421353, 0.29960036581881355], "threshold": 0.15}`
- **dev_oracle_gap_closure**: pass=dev-target — `{"oracle": 0.7220833333333333, "uniform": 0.5786, "curator": 0.6404333333333332, "gap_closure": 0.4309443605529088, "dev_target": 0.75, "note": "dev target, not pass/fail (60-75% needs documented analysis)"}`

**Overall: PASS**
