# Phase 2 Multi-Domain Split V2 Report

- Created: 2026-08-05T13:45:49
- Seed: `20260805`
- DFDC group ratios: train=0.6, val=0.2, test=0.1, holdout=remaining

## Output Splits

- train_v2: `D:\KyV_HocVienNganHang\NCKH\Final\Demo\splits\v2\train_v2.txt`
- val_v2: `D:\KyV_HocVienNganHang\NCKH\Final\Demo\splits\v2\val_v2.txt`
- test_v2: `D:\KyV_HocVienNganHang\NCKH\Final\Demo\splits\v2\test_v2.txt`
- dfdc_holdout_test: `D:\KyV_HocVienNganHang\NCKH\Final\Demo\splits\v2\dfdc_holdout_test.txt`
- dfdc_train_part: `D:\KyV_HocVienNganHang\NCKH\Final\Demo\splits\v2\dfdc_train_part.txt`
- dfdc_val_part: `D:\KyV_HocVienNganHang\NCKH\Final\Demo\splits\v2\dfdc_val_part.txt`
- dfdc_test_part: `D:\KyV_HocVienNganHang\NCKH\Final\Demo\splits\v2\dfdc_test_part.txt`

## Split Summary

| Split | Source | Samples | Real | Fake | Groups |
|---|---|---:|---:|---:|---:|
| train_v2 | Celeb-DF | 4104 | 1885 | 2219 | 1051 |
| train_v2 | DFDC | 1152 | 228 | 924 | 238 |
| train_v2 | FaceForensics++ | 14380 | 2557 | 11823 | 3231 |
| train_v2 | ALL | 19636 | 4670 | 14966 | 4520 |
| val_v2 | DFDC | 374 | 69 | 305 | 79 |
| val_v2 | FaceForensics++ | 1547 | 284 | 1263 | 360 |
| val_v2 | ALL | 1921 | 353 | 1568 | 439 |
| test_v2 | Celeb-DF | 393 | 233 | 160 | 148 |
| test_v2 | DFDC | 196 | 40 | 156 | 40 |
| test_v2 | FaceForensics++ | 1808 | 316 | 1492 | 403 |
| test_v2 | ALL | 2397 | 589 | 1808 | 591 |
| dfdc_holdout_test | DFDC | 191 | 40 | 151 | 40 |
| dfdc_holdout_test | ALL | 191 | 40 | 151 | 40 |
| dfdc_train_part | DFDC | 1152 | 228 | 924 | 238 |
| dfdc_train_part | ALL | 1152 | 228 | 924 | 238 |
| dfdc_val_part | DFDC | 374 | 69 | 305 | 79 |
| dfdc_val_part | ALL | 374 | 69 | 305 | 79 |
| dfdc_test_part | DFDC | 196 | 40 | 156 | 40 |
| dfdc_test_part | ALL | 196 | 40 | 156 | 40 |

## Video Group Overlap

| Left | Right | Overlap Groups |
|---|---|---:|
| train_v2 | val_v2 | 0 |
| train_v2 | test_v2 | 0 |
| train_v2 | dfdc_holdout_test | 0 |
| val_v2 | test_v2 | 0 |
| val_v2 | dfdc_holdout_test | 0 |
| test_v2 | dfdc_holdout_test | 0 |

## Notes

- Existing train/val/test samples are preserved and DFDC samples are added by video group.
- `dfdc_holdout_test.txt` must not be used during training or threshold selection.
- Old V1 splits are not overwritten.
