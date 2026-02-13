
import optuna
from optuna.trial import TrialState

def seed_trials():
    study_name = "onedrive_optimizer"
    storage_name = "sqlite:///optuna_study.db"
    
    # Create or load study
    study = optuna.create_study(
        study_name=study_name, 
        storage=storage_name, 
        load_if_exists=True, 
        direction='minimize'
    )
    
    current_trials = len(study.trials)
    target_start_trial = 22
    
    print(f"Current trial count: {current_trials}")
    
    if current_trials < target_start_trial:
        needed = target_start_trial - current_trials
        print(f"Seeding {needed} dummy trials to skip to Trial {target_start_trial}...")
        
        for i in range(needed):
            # Create a new trial
            trial = study.ask()
            # Mark it as FAIL so it doesn't affect the optimization results (best value)
            # We pass state=TrialState.FAIL. 
            # Note: tell() expects a value if state is COMPLETE, but None is fine for FAIL.
            study.tell(trial, state=TrialState.FAIL)
            
        print(f"Seeding complete. Next trial will be Trial {target_start_trial}.")
    else:
        print(f"Study already has {current_trials} trials. Next trial is {current_trials}.")

if __name__ == "__main__":
    seed_trials()
