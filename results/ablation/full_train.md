# Training on all BDD100K train frames (ablation c, extended)

69,998 frames, 861,718 objects, 10,313,532 detections (740,649 real); model 2 stopped at 142 iterations (adopted, subset: 107).

## Model 2 on all train frames vs the adopted model 2 (21,831-frame subset): cost ratio, 95% image-bootstrap

| dataset | group | lam | all train / subset | 95% CI |
|---|---|---|---|---|
| BDD100K | rain_night | 1 | 0.985 | 0.974-0.997 |
| BDD100K | rain_night | 3 | 0.994 | 0.984-1.005 |
| BDD100K | rain_night | 10 | 0.990 | 0.976-1.002 |
| BDD100K | rain_night | 30 | 0.985 | 0.971-0.998 |
| BDD100K | day | 1 | 0.992 | 0.989-0.995 |
| BDD100K | day | 3 | 0.992 | 0.989-0.995 |
| BDD100K | day | 10 | 0.989 | 0.986-0.992 |
| BDD100K | day | 30 | 0.992 | 0.988-0.996 |
| BDD100K | night | 1 | 0.990 | 0.987-0.993 |
| BDD100K | night | 3 | 0.992 | 0.989-0.995 |
| BDD100K | night | 10 | 0.990 | 0.986-0.993 |
| BDD100K | night | 30 | 0.990 | 0.987-0.994 |
| BDD100K | all | 1 | 0.991 | 0.989-0.993 |
| BDD100K | all | 3 | 0.992 | 0.990-0.994 |
| BDD100K | all | 10 | 0.990 | 0.987-0.992 |
| BDD100K | all | 30 | 0.991 | 0.988-0.993 |
| DAWN | fog | 1 | 1.025 | 1.001-1.052 |
| DAWN | fog | 3 | 1.012 | 0.987-1.041 |
| DAWN | fog | 10 | 1.009 | 0.985-1.033 |
| DAWN | fog | 30 | 1.007 | 0.984-1.030 |
| DAWN | all | 1 | 0.998 | 0.985-1.009 |
| DAWN | all | 3 | 1.007 | 0.996-1.018 |
| DAWN | all | 10 | 1.007 | 0.995-1.020 |
| DAWN | all | 30 | 1.009 | 0.998-1.021 |

## Cost vs one threshold at lam = 10

| dataset | group | one threshold from | per detection (model 2, all train) | 95% CI | model 1, all train |
|---|---|---|---|---|---|
| BDD100K | day | own (all train) | 0.908 | 0.902-0.913 | 0.931 |
| BDD100K | night | own (all train) | 0.895 | 0.889-0.901 | 0.934 |
| BDD100K | all | own (all train) | 0.901 | 0.897-0.906 | 0.932 |
| BDD100K | day | adopted (subset) | 0.908 | 0.902-0.913 | 0.931 |
| BDD100K | night | adopted (subset) | 0.894 | 0.888-0.901 | 0.933 |
| BDD100K | all | adopted (subset) | 0.901 | 0.896-0.905 | 0.932 |
| DAWN | fog | own (all train) | 0.961 | 0.928-0.995 | 0.983 |
| DAWN | all | own (all train) | 0.938 | 0.917-0.957 | 0.944 |
| DAWN | fog | adopted (subset) | 0.965 | 0.932-0.998 | 0.986 |
| DAWN | all | adopted (subset) | 0.940 | 0.920-0.959 | 0.946 |

## Calibration error (ECE, points)

| dataset | group | model 2, all train | model 2, subset (adopted) |
|---|---|---|---|
| BDD100K | all | 0.07 | 0.11 |
| BDD100K | day | 0.10 | 0.10 |
| BDD100K | night | 0.10 | 0.16 |
| DAWN | all | 1.03 | 0.98 |
| DAWN | fog | 1.59 | 1.49 |

## CRC over lambda (model 2, all train) vs per-weather CRC over the score, alpha 0.2, 200 splits

- conditions where the baseline holds: 11 of 11; ours max mean miss there 0.1981
- pooled false-alarm ratio ours / baseline: 0.587 (0.567-0.615); adopted model (subset): 0.610 (0.589-0.633)
