import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
scripts = [
    "01_eda.py",
    "02_normalization.py",
    "03_gt_validation.py",
    "04_positive_pairs.py",
    "05_blocking.py",
    "06_features.py",
    "07_train_model.py",
    "08_threshold.py",
    "09_output.py",
]

for script in scripts:
    print("\n" + "=" * 80)
    print("RUNNING", script)
    print("=" * 80)
    result = subprocess.run([sys.executable, str(ROOT / "scripts" / script)])
    if result.returncode != 0:
        print(f"\nSTOPPED: {script} failed with code {result.returncode}")
        sys.exit(result.returncode)

print("\nPipeline completed.")
