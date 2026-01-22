# COVS-runner

Deterministic runner for COVS verification (RUN A / RUN B).

This repository contains ONLY canonical, immutable execution artifacts.
No source code is executed directly from this repository.

## Contents

- run_a__covs__v1.0.1-canonical.zip  
  Canonical implementation of RUN A (OAR)

- run_b__covs__v1.0.2-canonical.zip  
  Canonical implementation of RUN B (OCN / OII / ONO)

- executor__covs__v1.0.2-canonical.zip  
  Dataset preparation + negative control (random walk)

- .github/workflows/  
  GitHub Actions pipeline for deterministic verification

## Execution Model

- ZIPs are NOT unpacked manually
- Execution is performed ONLY via CI
- All constants are preregistered
- All outputs are PASS / FAIL only
- No training, tuning, or optimization

## Verification

The green badge indicates that:
- All ZIP hashes are verified
- The negative control fails correctly
- RUN A and RUN B execute deterministically
- Results are reproducible

Any modification to ZIP files or workflow
INVALIDATES the verification.
