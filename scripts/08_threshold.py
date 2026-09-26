from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.metrics import precision_recall_curve

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT/"outputs"

df = pd.read_parquet(OUT/"scored_candidates.parquet")

# F0.5 = (1+beta^2)*P*R / (beta^2*P + R)
beta = 0.5

thresholds = np.linspace(0.05, 0.99, 95)
rows = []

for t in thresholds:
    pred = df["score"] >= t
    tp = int(((pred == 1) & (df["label"] == 1)).sum())
    fp = int(((pred == 1) & (df["label"] == 0)).sum())
    fn = int(((pred == 0) & (df["label"] == 1)).sum())

    precision = tp/(tp+fp) if tp+fp else 0
    recall = tp/(tp+fn) if tp+fn else 0
    f05 = (
        (1+beta**2)*precision*recall /
        (beta**2*precision + recall)
        if precision+recall else 0
    )

    rows.append([t,precision,recall,f05,tp,fp,fn])

result = pd.DataFrame(
    rows,
    columns=["threshold","precision","recall","f05","tp","fp","fn"]
)

result.to_csv(OUT/"threshold_search.csv", index=False)

best = result.loc[result["f05"].idxmax()]
print(best.to_string())
