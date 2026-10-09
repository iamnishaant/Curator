# CRE kappa/tau tuning (D-80)

Tuning seeds 0..49, S-I; objective = mean relative improvement over Uniform.

## cre

| # | params | objective | SE | norm. entropy | mean max-w | |
|---|---|---|---|---|---|---|
| 0 | {"exploration_coef": 0.02, "tau": 0.3} | -0.0024 | 0.0074 | 0.68 | 0.48 |  |
| 1 | {"exploration_coef": 0.02, "tau": 1.0} | +0.0058 | 0.0057 | 0.81 | 0.39 |  |
| 2 | {"exploration_coef": 0.05, "tau": 0.3} | -0.0041 | 0.0074 | 0.70 | 0.46 |  |
| 3 | {"exploration_coef": 0.05, "tau": 1.0} | +0.0124 | 0.0064 | 0.81 | 0.39 | cand. |
| 4 | {"exploration_coef": 0.1, "tau": 0.3} | +0.0122 | 0.0074 | 0.72 | 0.43 | cand. |
| 5 | {"exploration_coef": 0.1, "tau": 1.0} | +0.0047 | 0.0067 | 0.81 | 0.39 |  |
| 6 | {"exploration_coef": 0.2, "tau": 0.3} | -0.0051 | 0.0074 | 0.74 | 0.40 |  |
| 7 | {"exploration_coef": 0.2, "tau": 1.0} | +0.0098 | 0.0067 | 0.81 | 0.38 | **chosen** |

## curator_targeted

| # | params | objective | SE | norm. entropy | mean max-w | |
|---|---|---|---|---|---|---|
| 0 | {"exploration_coef": 0.02, "tau": 0.3} | -0.0272 | 0.0084 | 0.59 | 0.61 |  |
| 1 | {"exploration_coef": 0.02, "tau": 1.0} | -0.0163 | 0.0080 | 0.79 | 0.43 |  |
| 2 | {"exploration_coef": 0.05, "tau": 0.3} | -0.0183 | 0.0088 | 0.61 | 0.58 |  |
| 3 | {"exploration_coef": 0.05, "tau": 1.0} | -0.0097 | 0.0066 | 0.79 | 0.43 |  |
| 4 | {"exploration_coef": 0.1, "tau": 0.3} | -0.0143 | 0.0075 | 0.69 | 0.49 |  |
| 5 | {"exploration_coef": 0.1, "tau": 1.0} | -0.0128 | 0.0074 | 0.79 | 0.42 |  |
| 6 | {"exploration_coef": 0.2, "tau": 0.3} | -0.0058 | 0.0073 | 0.73 | 0.42 |  |
| 7 | {"exploration_coef": 0.2, "tau": 1.0} | +0.0029 | 0.0067 | 0.81 | 0.40 | **chosen** |
