"""Historical alias for the canonical SGSA model (``sgsa_utils/model.py``).

Earlier single-model scripts (``sgsa_train.py``, ``sgsa_evaluate.py``) imported
``model.SGSA``. This alias keeps them working without a second implementation.
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from sgsa_utils.model import SGSA, SimAM1D, load_historical_checkpoint  # noqa: E402

__all__ = ['SGSA', 'SimAM1D', 'load_historical_checkpoint']
