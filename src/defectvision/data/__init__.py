"""M2 - Data engineering for ML.

Ingestion, validation, feature extraction, splitting and the preprocessing /
augmentation pipeline. Nothing in this package imports torch except
:mod:`defectvision.data.dataset`, so the validation and statistics stages stay
cheap to run and easy to unit-test.
"""
