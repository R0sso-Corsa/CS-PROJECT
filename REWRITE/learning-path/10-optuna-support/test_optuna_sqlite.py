import optuna
from pathlib import Path
try:
    import sqlite3
    print("SQLite3 available")
    rewrite_root = next(p for p in Path(__file__).resolve().parents if p.name == "REWRITE")
    db_path = rewrite_root / "artifacts" / "legacy" / "databases" / "test_study.db"
    db_path.parent.mkdir(parents=True, exist_ok=True)
    storage_name = f"sqlite:///{db_path.as_posix()}"
    study = optuna.create_study(study_name="test_study", storage=storage_name, direction='minimize')
    print("Study created successfully with SQLite backend")
except Exception as e:
    print(f"Failed: {e}")
