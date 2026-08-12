"""Risk management: position sizing, drawdown guardrails, VaR, stop-loss."""

import sys
from pathlib import Path
SCRIPT_PATH = Path(__file__).resolve()
# Walk up from this file's location until we find the directory that contains 'src/'
PROJECT_ROOT = SCRIPT_PATH
while PROJECT_ROOT.parent != PROJECT_ROOT:
    if (PROJECT_ROOT / "src").is_dir():
        break
    PROJECT_ROOT = PROJECT_ROOT.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
from src.risk.position_sizing import PositionSizer, KellyCriterion
from src.risk.guardrails import RiskGuardrails, RiskReport

__all__ = ["PositionSizer", "KellyCriterion", "RiskGuardrails", "RiskReport"]
