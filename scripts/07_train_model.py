from pathlib import Path
import pandas as pd
import numpy as np
from lightgbm import LGBMClassifier

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT/"outputs"

features = pd.read_parquet(OUT/"candidate_features.parquet")

# This stage requires labels. The exact GT join is performed here.
# Since the candidate set is only the current blocker output, label
# positives from the full GT partitions.

positive = set()
for f in sorted((OUT/"positive_pairs_parts").glob("*.parquet")):
    p = pd.read_parquet(f)
    positive.update(
        zip(p.source1_entity_id.astype(str), p.matched_entity_id.astype(str))
    )

features["label"] = [
    int((a,b) in positive)
    for a,b in zip(features.source1_entity_id, features.matched_entity_id)
]

feature_cols = [
    "name_ratio",
    "name_wratio",
    "address_ratio",
    "same_country",
    "name_len_diff",
    "address_len_diff",
]

X = features[feature_cols]
y = features["label"]

if y.nunique() < 2:
    raise RuntimeError("Candidate set contains only one class. Improve blocking before training.")

model = LGBMClassifier(
    n_estimators=300,
    learning_rate=0.05,
    num_leaves=31,
    class_weight="balanced",
    random_state=42,
    n_jobs=-1
)

model.fit(X,y)
features["score"] = model.predict_proba(X)[:,1]

features.to_parquet(OUT/"scored_candidates.parquet", index=False)

import joblib
joblib.dump(model, OUT/"entity_match_lgbm.joblib")

print("Model trained.")
print("Positive candidates:", int(y.sum()))
print("Candidate rows:", len(y))
