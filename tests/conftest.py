import sys
from pathlib import Path

ROBOMIMIC_ROOT = Path(__file__).resolve().parents[1] / "third_party" / "robomimic"

sys.path.insert(0, str(ROBOMIMIC_ROOT))
