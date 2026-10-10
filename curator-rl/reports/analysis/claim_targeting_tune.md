# Calibration targeting by proxy claim, tuning (D-93)

Tuning seeds 0..49; refit scenarios S-I-R1, S-I-R2, S-J-R. Rule: false flags <= 0.05, calibration cost <= 0.1, score loss vs exposure/1 <= 0.005; pick the highest detection, ties within 0.02 to the cheaper; no change unless detection improves by >= 0.1.

| variant | mean score | noisy detected | false-flag rate | calibration cost | noisy weight | eligible | per-scenario detection |
|---|---|---|---|---|---|---|---|
| exposure/1 | 0.3208 | 0.267 | 0.017 | 0.023 | 0.222 | yes | S-I-R1 0.26, S-I-R2 0.20, S-J-R 0.34 |
| claim/1 | 0.3224 | 0.280 | 0.017 | 0.023 | 0.222 | yes | S-I-R1 0.32, S-I-R2 0.22, S-J-R 0.30 |
| claim/2 | 0.3232 | 0.727 | 0.022 | 0.034 | 0.195 | yes | S-I-R1 0.74, S-I-R2 0.74, S-J-R 0.70 |
| claim_stale/1 | 0.3236 | 0.500 | 0.020 | 0.023 | 0.212 | yes | S-I-R1 0.54, S-I-R2 0.50, S-J-R 0.46 |
| claim_stale/2 | 0.3258 | 0.780 | 0.017 | 0.034 | 0.188 | yes | S-I-R1 0.80, S-I-R2 0.76, S-J-R 0.78 |

**Chosen: claim_stale/2**
