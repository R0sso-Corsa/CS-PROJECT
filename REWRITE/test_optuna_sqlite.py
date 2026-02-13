import optuna
try:
    import sqlite3
    print("SQLite3 available")
    storage_name = "sqlite:///test_study.db"
    study = optuna.create_study(study_name="test_study", storage=storage_name, direction='minimize')
    print("Study created successfully with SQLite backend")
except Exception as e:
    print(f"Failed: {e}")
