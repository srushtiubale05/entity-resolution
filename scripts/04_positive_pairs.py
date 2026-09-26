from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
OUT = ROOT / "outputs"
OUT.mkdir(exist_ok=True)

# The complete GT is 7.6M pairs. Store it as partitioned Parquet
# instead of a giant in-memory DataFrame.

part_dir = OUT / "positive_pairs_parts"
part_dir.mkdir(exist_ok=True)

part = 0
total = 0

for gt in pd.read_csv(
    DATA/"train_ground_truth.tsv",
    sep="\t",
    dtype=str,
    usecols=["source1_entity_id","matched_entity_ids"],
    chunksize=50_000
):
    gt["matched_entity_id"] = gt["matched_entity_ids"].fillna("").str.split(",")
    pairs = gt[["source1_entity_id","matched_entity_id"]].explode("matched_entity_id")
    pairs["matched_entity_id"] = pairs["matched_entity_id"].str.strip()
    pairs = pairs[pairs["matched_entity_id"] != ""]
    pairs.to_parquet(part_dir / f"part_{part:05d}.parquet", index=False)
    total += len(pairs)
    part += 1

print(f"Created {part} positive-pair partitions.")
print(f"Total positive pairs: {total:,}")
