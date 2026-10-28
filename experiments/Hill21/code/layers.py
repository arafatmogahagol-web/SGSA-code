"""Historical alias for the canonical temporal SimAM module (``sgsa_utils/model.py``)."""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from sgsa_utils.model import SimAM1D  # noqa: E402

__all__ = ['SimAM1D']
