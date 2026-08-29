"""M5 - monitoring, observability and retraining.

The pieces fit together as a loop:

``store``            every served prediction is logged with the image statistics
                     that describe the input
``reference``        the training distribution is snapshotted once, as the
                     baseline "normal" to compare against
``drift``            statistical tests (PSI, KS, chi-square) quantify how far a
                     window of production traffic has moved from that baseline
``simulate_drift``   deliberately shifted inputs, so the detectors are proven to
                     fire before a real shift arrives
``report``           turns the signals into a readable report with plots
``retrain_trigger``  encodes when a signal is severe and persistent enough to
                     justify retraining
"""

from .drift import chi_square_test, ks_test, population_stability_index
from .store import PredictionRecord, PredictionStore

__all__ = [
    "PredictionRecord",
    "PredictionStore",
    "chi_square_test",
    "ks_test",
    "population_stability_index",
]
