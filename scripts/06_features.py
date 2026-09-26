from pathlib import Path
import pandas as pd
import numpy as np
from rapidfuzz.fuzz import ratio, WRatio

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs"

pairs_path = OUT/"candidate_pairs_exact_name.parquet"
if not pairs_path.exists():
    raise FileNotFoundError("Run 05_blocking.py first.")

pairs = pd.read_parquet(pairs_path)

s1 = pd.read_csv(ROOT/"data/train_source1.tsv", sep="\t", dtype=str).set_index("entity_id")
s2 = pd.read_csv(ROOT/"data/train_source2.tsv", sep="\t", dtype=str).set_index("entity_id")
s3 = pd.read_csv(ROOT/"data/train_source3.tsv", sep="\t", dtype=str).set_index("entity_id")

target = pd.concat([s2,s3])

def clean(x):
    return "" if pd.isna(x) else str(x).lower().strip()

rows = []

for _, p in pairs.iterrows():
    a = s1.loc[p.source1_entity_id]
    b = target.loc[p.matched_entity_id]

    an, bn = clean(a.business_name), clean(b.business_name)
    aa, ba = clean(a.business_address), clean(b.business_address)

    rows.append({
        "source1_entity_id": p.source1_entity_id,
        "matched_entity_id": p.matched_entity_id,
        "name_ratio": ratio(an,bn)/100,
        "name_wratio": WRatio(an,bn)/100,
        "address_ratio": ratio(aa,ba)/100 if aa and ba else 0,
        "same_country": int(clean(a.country) == clean(b.country)),
        "name_len_diff": abs(len(an)-len(bn)),
        "address_len_diff": abs(len(aa)-len(ba)),
    })

features = pd.DataFrame(rows)
features.to_parquet(OUT/"candidate_features.parquet", index=False)
print("Features:", features.shape)
