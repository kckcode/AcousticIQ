import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

from scripts.train_and_evaluate import run_experiment

for machine in ["pump", "fan", "slider"]:
    print(f"\n>>> Running Training & Evaluation for {machine}...")
    run_experiment(machine_category=machine, epochs=10, batch_size=16)
print("\n>>> All machine models successfully trained and evaluated!")
