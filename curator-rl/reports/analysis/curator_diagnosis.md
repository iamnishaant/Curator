# Curator v0 diagnosis (D-69)

Tuning seeds 0..39. Values are mean relative improvement over Uniform (paired by seed) with the standard error.

## 1. Component ablation

| variant | S-I | S-J | S-A | S-C |
|---|---|---|---|---|
| curator (as configured) | +0.0106 (0.0070) | +0.0336 (0.0097) | +0.1208 (0.0112) | +0.1783 (0.0181) |
| status soft | +0.0232 (0.0073) | +0.0626 (0.0078) | +0.1075 (0.0126) | +0.1783 (0.0181) |
| status off | +0.0208 (0.0087) | +0.0548 (0.0073) | +0.1046 (0.0137) | +0.1832 (0.0172) |
| status off, gamma 1.0 | +0.0240 (0.0079) | +0.0853 (0.0086) | +0.0938 (0.0168) | +0.1811 (0.0188) |
| status off, gamma 1.0, no cost (= Standard UCB form) | +0.0526 (0.0070) | +0.0781 (0.0085) | +0.0974 (0.0180) | +0.0478 (0.0189) |
| Standard UCB (tuned baseline) | +0.0526 (0.0070) | +0.0781 (0.0085) | +0.0974 (0.0180) | +0.0478 (0.0189) |

## 2. Spurious-reward upper bound (privileged S5 on the noisy arm, from round 6; NOT a method)

| variant | S-I | S-J | S-A |
|---|---|---|---|
| no S5 information | +0.0106 (0.0070), noisy share 0.25 | +0.0336 (0.0097), noisy share 0.22 | +0.1208 (0.0112), noisy share 0.20 |
| privileged S5, shrink 0.5 | +0.0302 (0.0083), noisy share 0.08 | +0.0779 (0.0074), noisy share 0.07 | +0.1580 (0.0062), noisy share 0.08 |
| privileged S5, shrink 0.1 | +0.0359 (0.0062), noisy share 0.06 | +0.0839 (0.0076), noisy share 0.06 | +0.1673 (0.0064), noisy share 0.04 |
