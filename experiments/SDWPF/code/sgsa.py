"""Re-export of the canonical SGSA model (see ``sgsa_utils/model.py``).

The two experiment protocols used to carry separate copies of this class. A
single implementation now lives in ``sgsa_utils/model.py``; this module only
keeps the historical import path working.
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from sgsa_utils.model import SGSA, SimAM1D, load_historical_checkpoint  # noqa: E402

__all__ = ['SGSA', 'SimAM1D', 'load_historical_checkpoint']
