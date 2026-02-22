
import optuna
import sys

# Point to the database used in onedrive_pytorch.py
db_url = "sqlite:///optuna_study.db"
study_name = "onedrive_optimizer"

try:
    study = optuna.load_study(study_name=study_name, storage=db_url)
    print(f"Loaded study '{study_name}' with {len(study.trials)} trials.")
except KeyError:
    print(f"Study '{study_name}' not found.")
    sys.exit(1)

print("\n--- Failed Trials Analysis ---")
failed_trials = [t for t in study.trials if t.state == optuna.trial.TrialState.FAIL]

if not failed_trials:
    print("No failed trials found.")
else:
    print(f"Found {len(failed_trials)} failed trials.")
    # Print details of the last failed trial, as it's likely the same error repeated
    last_failed = failed_trials[-1]
    print(f"\nLast Failed Trial (ID: {last_failed.number}):")
    print(f"Params: {last_failed.params}")
    print("System Attributes (often contains failure reason):")
    # Optuna stores exception info in system_attrs under 'fail_reason' key usually, depends on version
    # Actually, optuna doesn't always store the full traceback in a rigorous way in the trial object attributes unless configured.
    # However, let's inspect what we have.
    for k, v in last_failed.system_attrs.items():
        print(f"  {k}: {v}")
    
    # Sometimes it's useful to just try running the objective function if we can, 
    # but here we just want to see if Optuna recorded the exception.
    # In many setups, the exception is printed to stderr but not stored in DB.
    # Let's hope the user is seeing it in the terminal, but if not, we might need to reproduce it.

print("\n--- Trial States Summary ---")
states = {}
for t in study.trials:
    states[t.state] = states.get(t.state, 0) + 1
for s, count in states.items():
    print(f"{s}: {count}")
