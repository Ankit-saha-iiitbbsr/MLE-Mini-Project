"""M3 - model architectures and the factory that builds them from config.

Every deep model in this package emits a **single logit per image** rather than
a two-way softmax. For a binary defect check that is the more useful interface:
``sigmoid(logit)`` is directly P(defect), which is what the operating threshold
is tuned against, what gets logged for monitoring, and what the human-review
band is defined over. It also lets class imbalance be handled with a scalar
``pos_weight`` instead of a weight vector.
"""

from .baseline_cnn import AvgMaxPool, BaselineCNN
from .factory import ARCHITECTURES, build_model, count_parameters, describe_model
from .transfer import build_transfer_model

__all__ = [
    "ARCHITECTURES",
    "AvgMaxPool",
    "BaselineCNN",
    "build_model",
    "build_transfer_model",
    "count_parameters",
    "describe_model",
]
