"""Data layer: ingestion, feature engineering, and dataset contracts."""

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
from src.data.features import FeatureEngineer, FEATURE_COLS
from src.data.ingestion import DataIngestor

__all__ = ["FeatureEngineer", "FEATURE_COLS", "DataIngestor"]
