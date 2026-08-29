"""Feature extraction shared by the training pipeline and the live service.

The functions here are deliberately dependency-light (NumPy + Pillow only) so
that the serving container can compute the exact same values as the offline
pipeline. Reusing one implementation on both sides is the concrete defence
against training-serving skew (M2 / CS4).
"""

from .image_stats import (
    FEATURE_NAMES,
    image_statistics,
    statistics_frame,
    to_vector,
)

__all__ = ["FEATURE_NAMES", "image_statistics", "statistics_frame", "to_vector"]
