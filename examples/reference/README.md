# Reference

## Purpose

Measurement-grade offline golden harness for `CalibrationArtifact` behavior.

## Expected input

`reproduce_reference_artifact.py` builds deterministic in-memory datasets,
rubrics, variants, providers, and judges. It does not need API keys or network
access.

## Expected output

`reference_artifact.json` plus golden variants for KC4 failure, hard-gate
failure, provider degradation, and diff regression. `golden_manifest.json`
records expected status, ship recommendation, validation mode, integrity
classification, and normalized artifact hash.
