import sys
from pathlib import Path

TESTS_ROOT = Path(__file__).resolve().parent
ROBOMIMIC_ROOT = TESTS_ROOT.parent / "third_party" / "robomimic"

sys.path.insert(0, str(ROBOMIMIC_ROOT))
sys.path.insert(0, str(TESTS_ROOT))
