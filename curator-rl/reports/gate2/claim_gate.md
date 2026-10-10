# H-claim: calibration targeting claim_stale/2 (D-94, pre-registered)

Generated 20261010T034517Z; fresh seeds 600..699; refit scenarios; candidate = base config with calib.targeting claim_stale, max_targets 2.

## Mean final scores

| scenario | uniform | lp | ucb | sec | dump | curator | curator_cs2 | curator_g1_cs2 |
|---|---|---|---|---|---|---|---|---|
| S-I-R1 | 0.2912 | 0.3294 | 0.3288 | 0.2925 | 0.2947 | 0.3176 | 0.3238 | 0.3516 |
| S-I-R2 | 0.2913 | 0.3284 | 0.3256 | 0.2919 | 0.2940 | 0.3218 | 0.3253 | 0.3321 |
| S-J-R | 0.2908 | 0.3363 | 0.3264 | 0.2901 | 0.2922 | 0.3198 | 0.3258 | 0.3299 |

## Primary criteria

- **C1_detection**: PASS
  - S-I-R1: +0.6100 [+0.5100, +0.7100] (detection 0.86 vs 0.25) -> pass
  - S-I-R2: +0.4900 [+0.3800, +0.6000] (detection 0.81 vs 0.32) -> pass
  - S-J-R: +0.4700 [+0.3500, +0.5800] (detection 0.78 vs 0.31) -> pass
- **C2_safety**: PASS
  - S-I-R1: false-flag 0.020, calibration cost 0.034 -> pass
  - S-I-R2: false-flag 0.025, calibration cost 0.033 -> pass
  - S-J-R: false-flag 0.022, calibration cost 0.033 -> pass
- **C3_non_inferiority**: PASS
  - S-I-R1: +0.0061 [+0.0021, +0.0103] -> pass
  - S-I-R2: +0.0035 [-0.0002, +0.0074] -> pass
  - S-J-R: +0.0060 [+0.0018, +0.0102] -> pass
- **C4_gate2_retained**: PASS
  - S-I-R1:cs2-sec: +0.0312 [+0.0268, +0.0359] -> pass
  - S-I-R1:cs2-dump: +0.0291 [+0.0244, +0.0338] -> pass
  - S-I-R2:cs2-sec: +0.0334 [+0.0291, +0.0378] -> pass
  - S-I-R2:cs2-dump: +0.0312 [+0.0268, +0.0355] -> pass
  - S-J-R:cs2-uniform: +0.0350 [+0.0298, +0.0400] -> pass

**Adopt claim_stale/2: YES**

## Secondary: score of the candidate minus Curator (exposure/1), Standard UCB, LP

| scenario | curator | ucb | lp |
|---|---|---|---|
| S-I-R1 | +0.0061 [+0.0022, +0.0103] | -0.0051 [-0.0111, +0.0010] | -0.0056 [-0.0099, -0.0013] |
| S-I-R2 | +0.0035 [-0.0003, +0.0074] | -0.0004 [-0.0065, +0.0058] | -0.0031 [-0.0079, +0.0018] |
| S-J-R | +0.0060 [+0.0017, +0.0102] | -0.0006 [-0.0073, +0.0061] | -0.0105 [-0.0155, -0.0056] |

## Exploratory (no decision attached): candidate with gamma = 1

| scenario | minus candidate (gamma 0.95) | minus Standard UCB |
|---|---|---|
| S-I-R1 | +0.0278 [+0.0214, +0.0341] | +0.0228 [+0.0156, +0.0301] |
| S-I-R2 | +0.0068 [+0.0022, +0.0115] | +0.0065 [-0.0001, +0.0134] |
| S-J-R | +0.0041 [-0.0011, +0.0093] | +0.0035 [-0.0030, +0.0102] |

## Detail

| scenario | method | noisy S5 | false-flag | calibration cost | noisy budget share |
|---|---|---|---|---|---|
| S-I-R1 | curator | 0.25 | 0.015 | 0.023 | 0.229 |
| S-I-R1 | curator_cs2 | 0.86 | 0.020 | 0.034 | 0.187 |
| S-I-R1 | curator_g1_cs2 | 0.74 | 0.065 | 0.040 | 0.177 |
| S-I-R2 | curator | 0.32 | 0.010 | 0.022 | 0.226 |
| S-I-R2 | curator_cs2 | 0.81 | 0.025 | 0.033 | 0.190 |
| S-I-R2 | curator_g1_cs2 | 0.75 | 0.050 | 0.036 | 0.207 |
| S-J-R | curator | 0.31 | 0.015 | 0.022 | 0.225 |
| S-J-R | curator_cs2 | 0.78 | 0.022 | 0.033 | 0.191 |
| S-J-R | curator_g1_cs2 | 0.66 | 0.030 | 0.033 | 0.194 |
