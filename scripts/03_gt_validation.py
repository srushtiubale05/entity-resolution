from pathlib import Path
import pandas as pd
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
OUT = ROOT / "outputs"
OUT.mkdir(exist_ok=True)

print("="*80)
print("PHASE 3 — GROUND TRUTH VALIDATION")
print("="*80)

s1_ids = set()
for df in pd.read_csv(DATA/"train_source1.tsv", sep="\t", dtype=str, usecols=["entity_id"], chunksize=200_000):
    s1_ids.update(df["entity_id"].dropna().astype(str))

s2_ids = set()
for df in pd.read_csv(DATA/"train_source2.tsv", sep="\t", dtype=str, usecols=["entity_id"], chunksize=200_000):
    s2_ids.update(df["entity_id"].dropna().astype(str))

s3_ids = set()
for df in pd.read_csv(DATA/"train_source3.tsv", sep="\t", dtype=str, usecols=["entity_id"], chunksize=200_000):
    s3_ids.update(df["entity_id"].dropna().astype(str))

print("Source IDs:", len(s1_ids), len(s2_ids), len(s3_ids))

joinable = 0
pairs = 0
s1_present = 0

for gt in pd.read_csv(
    DATA/"train_ground_truth.tsv",
    sep="\t", dtype=str,
    usecols=["source1_entity_id","matched_entity_ids"],
    chunksize=100_000
):
    for s1id, matches in zip(gt["source1_entity_id"], gt["matched_entity_ids"]):
        s1ok = str(s1id) in s1_ids
        if s1ok:
            s1_present += 1
        if pd.isna(matches):
            continue
        for mid in str(matches).split(","):
            mid = mid.strip()
            if not mid:
                continue
            pairs += 1
            if s1ok and (mid in s2_ids or mid in s3_ids):
                joinable += 1

print("GT positive pairs:", pairs)
print("S1 rows present:", s1_present)
print("Fully joinable pairs:", joinable)
print("Joinability:", f"{100*joinable/pairs:.4f}%" if pairs else "0")

pd.DataFrame([{
    "source1_ids": len(s1_ids),
    "source2_ids": len(s2_ids),
    "source3_ids": len(s3_ids),
    "gt_positive_pairs": pairs,
    "fully_joinable_pairs": joinable,
    "joinability_pct": 100*joinable/pairs if pairs else 0
}]).to_csv(OUT/"gt_validation.csv", index=False)
