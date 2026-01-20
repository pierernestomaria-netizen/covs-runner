# CoVS-runner

Deterministic runners for COVS verification (RUN A / RUN B).

This repository contains the **exact, reproducible execution runners**
used for the COVS (Criterion of Operational Structural Truth) verification.

There is **no training, tuning, or optimization**.
The runners are deterministic and produce binary outcomes (PASS / FAIL).
## Files

- `run_a_nasa.py`
RUN A — Empirical compatibility check on NASA C-MAPSS FD001.

- `run_b_lnpr.py`
RUN B — Structural falsification check (LNPR).
## Requirements

- Python >= 3.9 (tested with 3.11)
- ASCII-only environment
- No GPU required
- No external services
- No randomness (fully deterministic)
## Dataset (NASA C-MAPSS)

RUN A and RUN B are designed to operate on the **NASA C-MAPSS FD001 dataset**.

The dataset must be obtained from the official NASA source
and placed locally by the user.

Expected files (example):

- `train_FD001.txt`
- `test_FD001.txt`
- `RUL_FD001.txt`

The dataset is **NOT included** in this repository.
## Execution

### RUN A — NASA C-MAPSS

```bash
python run_a_nasa.py \
--train train_FD001.txt \
--test test_FD001.txt \
--rul RUL_FD001.txt
