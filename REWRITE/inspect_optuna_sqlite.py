
import sqlite3
import json

db_path = "optuna_study.db"

try:
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    # Check tables
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
    tables = [row[0] for row in cursor.fetchall()]
    print(f"Tables: {tables}")

    if 'trials' in tables:
        # Get failed trials
        # Optuna TrialState: FAIL=5 (usually). But let's just list all and their states.
        cursor.execute("SELECT trial_id, state, datetime_start, datetime_complete FROM trials WHERE state='FAIL' OR state=5")
        failed_trials = cursor.fetchall()
        print(f"\nFailed Trials Count: {len(failed_trials)}")
        
        if failed_trials:
            # Let's check system attributes for the last failed trial
            last_failed_id = failed_trials[-1][0]
            print(f"\nAnalyzing Last Failed Trial (ID: {last_failed_id})")
            
            # Check trial_system_attributes for fail reason
            if 'trial_system_attributes' in tables:
                cursor.execute("SELECT key, value_json FROM trial_system_attributes WHERE trial_id=?", (last_failed_id,))
                attrs = cursor.fetchall()
                print("System Attributes:")
                for k, v in attrs:
                    print(f"  {k}: {v}")
            
            # Check trial_user_attributes if any
            if 'trial_user_attributes' in tables:
                cursor.execute("SELECT key, value_json FROM trial_user_attributes WHERE trial_id=?", (last_failed_id,))
                attrs = cursor.fetchall()
                print("User Attributes:")
                for k, v in attrs:
                    print(f"  {k}: {v}")

    else:
        print("No 'trials' table found. Schema might be different (Optuna v2 vs v3?).")
        # Try finding any table with 'trial' in name
        
    conn.close()

except Exception as e:
    print(f"Error inspecting DB: {e}")
