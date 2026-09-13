# Current State

Last updated: 2026-09-07

## Current project

Two main forecasting approaches are retained:

1. Physics-based:
   ECMWF → CNN → VIC → RVIC → LSTM

2. Data-driven:
   FutureTST

## Documentation

Authoritative:
docs/PROJECT_DOCUMENTATION.md

Research history:
docs/FINDINGS.md

Historical documentation:
archive/historical_docs/

## Important

Do not quote results from archived documentation.
Current numerical results must come from the primary
documentation and current result artifacts.

## Development status

Core pipeline: implemented
CNN downscaling: implemented
VIC/RVIC: implemented
LSTM post-processing: implemented
FutureTST: implemented
Evaluation: implemented

FutureTST results currently flagged for re-run after
the date-label repair.
