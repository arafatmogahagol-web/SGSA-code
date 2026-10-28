"""Historical alias for the canonical SGSA model (``sgsa_utils/model.py``).

The Hill of Towie protocol passes its own turbine and feature counts
(``SGSA(96, 21, 7, ...)``) to the shared class; this module no longer carries a
dataset-specific copy that could disagree with the SDWPF protocol.
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from sgsa_utils.model import SGSA, SimAM1D, load_historical_checkpoint  # noqa: E402

__all__ = ['SGSA', 'SimAM1D', 'load_historical_checkpoint']
