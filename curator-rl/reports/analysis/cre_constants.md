# CRE constants frozen from tuning seeds (D-77)

Tuning seeds 0..49, scenarios S-A, S-B, S-C, S-I; free calibration, every slice, 400 paired items. 840 (seed, arm) residuals.

- `cre.proxy_scale` (s) = **0.0001963** (median of b_hat*c/xbar over 840 arm estimates)
- `cre.prior_rel_sd` (rho) = **1.232** (raw robust SD 1.232, clipped to (0.3, 3.0); median relative residual -0.000)
- `cre.roi_scale` (R_max) = **1.5105** (95th percentile of b_hat * B over 683 positive arm estimates)

| scenario | arm estimates | median b_hat*c/xbar | p95 b_hat * B |
|---|---|---|---|
| S-A | 123 | 0.00022537698282397588 | 0.6610582596113587 |
| S-B | 326 | 0.0002559333567672345 | 1.0075749445938982 |
| S-C | 130 | 0.0003916448579199956 | 1.5559928355462183 |
| S-I | 261 | 0.0001275843239026733 | 1.7629587605150705 |
