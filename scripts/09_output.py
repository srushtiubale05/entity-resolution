from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT/"outputs"

df = pd.read_parquet(OUT/"scored_candidates.parquet")
thresholds = pd.read_csv(OUT/"threshold_search.csv")
threshold = float(thresholds.loc[thresholds["f05"].idxmax(),"threshold"])

pred = df[df["score"] >= threshold].copy()
pred = pred.sort_values(
    ["source1_entity_id","score"],
    ascending=[True,False]
)

pred[[
    "source1_entity_id",
    "matched_entity_id",
    "score"
]].to_csv(
    OUT/"predicted_matches.tsv",
    sep="\t",
    index=False
)

print("Threshold:", threshold)
print("Predicted pairs:", len(pred))
print("Saved:", OUT/"predicted_matches.tsv")
