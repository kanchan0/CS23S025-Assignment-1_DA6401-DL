import wandb

ENTITY = "cs23s025-indian-institute-of-technology-madras"  
PROJECT = "CS23S025-Assignment-1-DA6401-DL"  
API = wandb.Api()

runs = API.runs(f"{ENTITY}/{PROJECT}")
sweep_dict = {}

for run in runs:
    if run.sweep:  # Check if the run belongs to a sweep
        sweep_id = run.sweep.id
        sweep_name = run.sweep.config.get("name", "Unnamed Sweep")
        sweep_dict[sweep_id] = sweep_name

for sweep_id, sweep_name in sweep_dict.items():
    print(f"Sweep Name: {sweep_name}, Sweep ID: {sweep_id}")
