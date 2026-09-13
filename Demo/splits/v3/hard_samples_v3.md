# Hard Sample Mining Report

- Created: 2026-08-06T10:10:48
- Model: `models\best_pytorch_model_b4_v2_packaged.pth`
- Split: `splits\v3\train_v3.txt`
- Fake FN-risk rule: `label=1 and prob_fake <= 0.5`
- Real FP-risk rule: `label=0 and prob_fake >= 0.5`
- Total predictions: `19926`
- Hard samples: `711`

## Summary

| Hard type | Source | Label | Samples |
|---|---|---:|---:|
| hard_fake_false_negative_risk | Celeb-DF | 1 | 5 |
| hard_fake_false_negative_risk | DFDC | 1 | 8 |
| hard_fake_false_negative_risk | FaceForensics++ | 1 | 572 |
| hard_real_false_positive_risk | Celeb-DF | 0 | 6 |
| hard_real_false_positive_risk | DFDC | 0 | 5 |
| hard_real_false_positive_risk | FaceForensics++ | 0 | 115 |

## Notes

- Use this CSV only with training/pool data, not held-out test data.
- Hard Fake samples are oversampled to reduce false negatives.
- Hard Real samples are also oversampled to protect precision while recall is improved.
