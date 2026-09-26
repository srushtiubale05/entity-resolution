
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
OUT = ROOT / "outputs"
OUT.mkdir(exist_ok=True)

files = {
    "source1": DATA / "train_source1.tsv",
    "source2": DATA / "train_source2.tsv",
    "source3": DATA / "train_source3.tsv",
    "gt": DATA / "train_ground_truth.tsv",
}

print("=" * 80)
print("PHASE 1 — MEMORY-SAFE EDA")
print("=" * 80)


for name, path in files.items():
    if not path.exists():
        raise FileNotFoundError(f"Missing: {path}")


def inspect_source(path, name, chunksize=200_000):
    total_rows = 0
    country_counts = {}
    missing_counts = {
        "business_name": 0,
        "business_address": 0,
        "country": 0,
    }

    first_chunk = None

    usecols = [
        "entity_id",
        "business_name",
        "business_address",
        "country",
    ]

    for chunk in pd.read_csv(
        path,
        sep="\t",
        dtype=str,
        usecols=usecols,
        chunksize=chunksize,
    ):
        if first_chunk is None:
            first_chunk = chunk.head(3).copy()

        total_rows += len(chunk)

        countries = chunk["country"].fillna("").value_counts()

        for country, count in countries.items():
            country_counts[country] = (
                country_counts.get(country, 0) + int(count)
            )

        for col in missing_counts:
            missing_counts[col] += int(chunk[col].isna().sum())

        del chunk

    print(f"\n{name}: rows={total_rows:,}")
    print("Columns:", usecols)

    print("\nFirst 3 rows:")
    print(first_chunk.to_string(index=False))

    print("\nCountry:")
    print(
        pd.Series(country_counts)
        .sort_values(ascending=False)
        .head(20)
        .to_string()
    )

    print("\nMissing:")
    print(pd.Series(missing_counts).to_string())

    return total_rows


counts = {}

counts["source1"] = inspect_source(
    files["source1"],
    "Source 1"
)

counts["source2"] = inspect_source(
    files["source2"],
    "Source 2"
)

counts["source3"] = inspect_source(
    files["source3"],
    "Source 3"
)


# ============================================================================
# GROUND TRUTH
# ============================================================================

print("\n" + "=" * 80)
print("GROUND TRUTH")
print("=" * 80)

gt_rows = 0
positive_references = 0
empty_match_rows = 0
gt_sample = None

gt_usecols = [
    "source1_entity_id",
    "matched_entity_ids",
]

for chunk in pd.read_csv(
    files["gt"],
    sep="\t",
    dtype=str,
    usecols=gt_usecols,
    chunksize=200_000,
):

    if gt_sample is None:
        gt_sample = chunk.head(20).copy()

    gt_rows += len(chunk)

    def count_matches(x):
        if pd.isna(x) or not str(x).strip():
            return 0

        return len(
            [z for z in str(x).split(",") if z.strip()]
        )

    parsed = chunk["matched_entity_ids"].map(count_matches)

    positive_references += int(parsed.sum())
    empty_match_rows += int((parsed == 0).sum())

    del chunk


print(f"GT rows: {gt_rows:,}")
print("GT columns:", gt_usecols)
print(f"Parsed positive references: {positive_references:,}")
print(f"GT rows with zero matches: {empty_match_rows:,}")

summary = pd.DataFrame([{
    "source1_rows": counts["source1"],
    "source2_rows": counts["source2"],
    "source3_rows": counts["source3"],
    "gt_rows": gt_rows,
    "positive_references": positive_references,
    "gt_rows_with_zero_matches": empty_match_rows,
}])

summary.to_csv(
    OUT / "eda_summary.csv",
    index=False
)

if gt_sample is not None:
    gt_sample.to_csv(
        OUT / "gt_sample.csv",
        index=False
    )

print("\nEDA complete.")
print("Saved:")
print("  outputs/eda_summary.csv")
print("  outputs/gt_sample.csv")
