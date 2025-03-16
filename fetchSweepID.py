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

'''
python3 fetchSweepID.py 

Sweep Name: DL-Assignment-1, Sweep ID: 38qy9h0x
Sweep Name: DL-Assignment-1_running_Remote, Sweep ID: xm7jkopc
Sweep Name: DL-Assignment-1_remote_SqLoss, Sweep ID: amu7xlj0
Sweep Name: DL-Assignment-1_sweep2, Sweep ID: sluhp8ra
Sweep Name: DL-Assignment-1_remote_SqLoss_sweep2, Sweep ID: oc3w4kuz
Sweep Name: DL-Assignment-1_sqLoss_sweep1, Sweep ID: 8z40b4y0
Sweep Name: DL-Assignment-1_sweep3, Sweep ID: sejxf76k
Sweep Name: DL-Assignment-1_sweep4, Sweep ID: nf3mjrmg
Sweep Name: DL-Assignment-1_finalSQloss, Sweep ID: 6xn5laor
Sweep Name: DL-Assignment-1_Sweep-5, Sweep ID: wawyr66w
Sweep Name: DL-Assignment-1_Sweep-6, Sweep ID: ah52mmyl
Sweep Name: DL-Assignment-1_Sweep-7, Sweep ID: ivyve8tl
Sweep Name: DL-Assignment-1_Sweep-8, Sweep ID: ebd3n6gv
'''