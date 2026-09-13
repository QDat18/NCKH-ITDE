# Multi-Domain Split V3 Report

- Created: 2026-08-06T10:04:33
- Seed: `20260806`
- Source pool: `train_v2 + val_v2` only.
- `test_v2` and `dfdc_holdout_test` are copied unchanged and must stay held out.

## Output Splits

- train_v3: `D:\KyV_HocVienNganHang\NCKH\Final\Demo\splits\v3\train_v3.txt`
- val_v3: `D:\KyV_HocVienNganHang\NCKH\Final\Demo\splits\v3\val_v3.txt`
- test_v3: `D:\KyV_HocVienNganHang\NCKH\Final\Demo\splits\v3\test_v3.txt`
- dfdc_holdout_test: `D:\KyV_HocVienNganHang\NCKH\Final\Demo\splits\v3\dfdc_holdout_test.txt`

## Validation Targets

| Source | Label | Target samples | Selected samples | Selected groups | Available groups |
|---|---:|---:|---:|---:|---:|
| FaceForensics++ | 0 | 300 | 304 | 68 | 636 |
| FaceForensics++ | 1 | 300 | 300 | 68 | 2955 |
| Celeb-DF | 0 | 300 | 303 | 17 | 124 |
| Celeb-DF | 1 | 300 | 300 | 132 | 927 |
| DFDC | 0 | 120 | 122 | 25 | 61 |
| DFDC | 1 | 300 | 302 | 64 | 256 |

## Split Summary

| Split | Source | Samples | Real | Fake | Groups |
|---|---|---:|---:|---:|---:|
| train_v3 | Celeb-DF | 3501 | 1582 | 1919 | 902 |
| train_v3 | DFDC | 1102 | 175 | 927 | 228 |
| train_v3 | FaceForensics++ | 15323 | 2537 | 12786 | 3455 |
| train_v3 | ALL | 19926 | 4294 | 15632 | 4585 |
| val_v3 | Celeb-DF | 603 | 303 | 300 | 149 |
| val_v3 | DFDC | 424 | 122 | 302 | 89 |
| val_v3 | FaceForensics++ | 604 | 304 | 300 | 136 |
| val_v3 | ALL | 1631 | 729 | 902 | 374 |
| test_v3 | Celeb-DF | 393 | 233 | 160 | 148 |
| test_v3 | DFDC | 196 | 40 | 156 | 40 |
| test_v3 | FaceForensics++ | 1808 | 316 | 1492 | 403 |
| test_v3 | ALL | 2397 | 589 | 1808 | 591 |
| dfdc_holdout_test | DFDC | 191 | 40 | 151 | 40 |
| dfdc_holdout_test | ALL | 191 | 40 | 151 | 40 |

## Video Group Overlap

| Left | Right | Overlap groups |
|---|---|---:|
| train_v3 | val_v3 | 0 |
| train_v3 | test_v3 | 0 |
| train_v3 | dfdc_holdout_test | 0 |
| val_v3 | test_v3 | 0 |
| val_v3 | dfdc_holdout_test | 0 |
| test_v3 | dfdc_holdout_test | 0 |

