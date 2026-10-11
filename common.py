"""
Pieces shared by train.py and app.py: file locations, the business-cost
assumptions, and the definition of the standard PyTorch model (app.py needs
the same architecture to load the saved weights back in).
"""

from pathlib import Path

import numpy as np
import torch.nn as nn

ROOT = Path(__file__).resolve().parent
DATA_PATH = ROOT / "data" / "noshows.csv"
ARTIFACTS = ROOT / "artifacts"

# --- Business framing: the clinic phones patients flagged as likely no-shows.
# Illustrative cost assumptions (per appointment), used both to pick the
# decision threshold in train.py and as the defaults in the dashboard.
CALL_COST = 2.0          # staff time for one reminder call
MISSED_SLOT_COST = 25.0  # an empty slot: idle doctor time + a patient not seen
CALL_EFFECT = 0.30       # share of would-be no-shows a call turns into a visit


def business_cost(y_true, call, call_cost=CALL_COST,
                  missed_slot_cost=MISSED_SLOT_COST, call_effect=CALL_EFFECT):
    """Total cost of a calling policy.

    y_true: 1 = no-show, 0 = attended. call: 1 = we phoned this patient.
    Every call costs `call_cost`. A no-show we didn't call costs a full
    missed slot; a no-show we did call only misses with prob (1 - call_effect).
    """
    y_true = np.asarray(y_true, dtype=float)
    call = np.asarray(call, dtype=float)
    expected_missed = y_true * (1.0 - call * call_effect)
    return float((call * call_cost).sum() + (expected_missed * missed_slot_cost).sum())


def build_standard_model(n_features):
    """Logistic regression as an nn.Module: linear layer followed by a sigmoid."""
    return nn.Sequential(nn.Linear(n_features, 1), nn.Sigmoid())
