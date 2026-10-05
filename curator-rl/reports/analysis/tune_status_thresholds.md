# Status threshold tuning study (Phase 4)

Truth anchored to the frozen base.yaml default thresholds (D-39).
Constraint: flip_rate <= 5 per 100 rounds. Objective: macro-F1.


### tuning: top 5 grid points

| rank | overrides | macro-F1 | flip rate |
|---|---|---|---|
| 1 | `{"p_sat": 0.7, "p_hard": 0.05, "z_up": 1.5, "dwell_min": 3, "consecutive_rounds": 4}` | 0.9705 | 0.37 |
| 2 | `{"p_sat": 0.7, "p_hard": 0.05, "z_up": 1.5, "dwell_min": 5, "consecutive_rounds": 4}` | 0.9705 | 0.37 |
| 3 | `{"p_sat": 0.7, "p_hard": 0.05, "z_up": 1.5, "dwell_min": 8, "consecutive_rounds": 4}` | 0.9705 | 0.37 |
| 4 | `{"p_sat": 0.7, "p_hard": 0.05, "z_up": 2.0, "dwell_min": 3, "consecutive_rounds": 4}` | 0.9705 | 0.37 |
| 5 | `{"p_sat": 0.7, "p_hard": 0.05, "z_up": 2.0, "dwell_min": 5, "consecutive_rounds": 4}` | 0.9705 | 0.37 |

### evaluation: top 5 grid points

| rank | overrides | macro-F1 | flip rate |
|---|---|---|---|
| 1 | `{"p_sat": 0.7, "p_hard": 0.05, "z_up": 1.5, "dwell_min": 3, "consecutive_rounds": 2}` | 0.9672 | 0.38 |
| 2 | `{"p_sat": 0.7, "p_hard": 0.05, "z_up": 1.5, "dwell_min": 5, "consecutive_rounds": 2}` | 0.9672 | 0.38 |
| 3 | `{"p_sat": 0.7, "p_hard": 0.05, "z_up": 1.5, "dwell_min": 8, "consecutive_rounds": 2}` | 0.9672 | 0.38 |
| 4 | `{"p_sat": 0.7, "p_hard": 0.05, "z_up": 2.0, "dwell_min": 3, "consecutive_rounds": 2}` | 0.9672 | 0.38 |
| 5 | `{"p_sat": 0.7, "p_hard": 0.05, "z_up": 2.0, "dwell_min": 5, "consecutive_rounds": 2}` | 0.9672 | 0.38 |