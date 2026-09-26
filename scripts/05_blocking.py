
from pathlib import Path
import pandas as pd
import re
import shutil
import gc

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs"

S2_DIR = OUT / "source2_norm_parts"
S3_DIR = OUT / "source3_norm_parts"

CAND_DIR = OUT / "candidate_pair_parts"
INDEX_DIR = OUT / "blocking_index"

# ---------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------

MAX_BLOCK_SIZE = 200
WRITE_BATCH_SIZE = 10_000

KEY_COLUMNS = [
    "block_country_name",
    "block_country_prefix",
    "block_country_signature",
    "block_country_address",
    "block_name_address",
]


# ---------------------------------------------------------------------
# Clean previous failed blocking output
# ---------------------------------------------------------------------

if CAND_DIR.exists():
    shutil.rmtree(CAND_DIR)

if INDEX_DIR.exists():
    shutil.rmtree(INDEX_DIR)

CAND_DIR.mkdir(parents=True, exist_ok=True)
INDEX_DIR.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------

def clean(value):
    if pd.isna(value):
        return ""

    value = str(value).lower().strip()
    value = re.sub(r"[^\w\s]", " ", value, flags=re.UNICODE)
    value = re.sub(r"\s+", " ", value).strip()

    return value


def name_prefix(value):
    value = clean(value).replace(" ", "")

    if len(value) < 4:
        return ""

    return value[:4]


def name_signature(value):
    value = clean(value).replace(" ", "")

    if len(value) < 5:
        return ""

    return "".join(sorted(set(value)))[:20]


def address_prefix(value):
    value = clean(value).replace(" ", "")

    if len(value) < 5:
        return ""

    return value[:5]


def add_keys(df):

    country = df["country"].fillna("").map(clean)
    name = df["business_name_norm"].fillna("")
    address = df["business_address_norm"].fillna("")

    df = df.copy()

    clean_name = name.map(clean)
    nprefix = name.map(name_prefix)
    nsig = name.map(name_signature)
    aprefix = address.map(address_prefix)

    df["block_country_name"] = (
        country + "|" + clean_name
    )

    df["block_country_prefix"] = (
        country + "|" + nprefix
    )

    df["block_country_signature"] = (
        country + "|" + nsig
    )

    df["block_country_address"] = (
        country + "|" + aprefix
    )

    df["block_name_address"] = (
        country
        + "|"
        + nprefix
        + "|"
        + aprefix
    )

    return df


# ---------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------

print("=" * 80)
print("PHASE 5 — LOW-MEMORY MULTI-KEY BLOCKING")
print("=" * 80)

s2_parts = sorted(S2_DIR.glob("*.parquet"))
s3_parts = sorted(S3_DIR.glob("*.parquet"))

if not s2_parts:
    raise FileNotFoundError(
        f"No Source 2 normalized partitions found: {S2_DIR}"
    )

if not s3_parts:
    raise FileNotFoundError(
        f"No Source 3 normalized partitions found: {S3_DIR}"
    )

print(f"Source 2 partitions: {len(s2_parts)}")
print(f"Source 3 partitions: {len(s3_parts)}")


# ---------------------------------------------------------------------
# STEP 1
# Build Source 3 blocking maps ON DISK.
#
# Instead of one giant dictionary, each source3 partition becomes a
# small collection of key -> IDs files.
# ---------------------------------------------------------------------

def build_source3_index():

    print("\nBuilding Source 3 blocking indexes...")

    index_files = []

    for i, path in enumerate(s3_parts):

        df = pd.read_parquet(path)

        df = add_keys(df)

        keep = [
            "entity_id",
            *KEY_COLUMNS,
        ]

        df = df[keep]

        out = INDEX_DIR / f"s3_{i:04d}.parquet"

        df.to_parquet(
            out,
            index=False,
            compression="snappy",
        )

        index_files.append(out)

        del df
        gc.collect()

        print(
            f"  S3 index "
            f"{i + 1}/{len(s3_parts)}"
        )

    return index_files


s3_index = build_source3_index()


# ---------------------------------------------------------------------
# Candidate writer
# ---------------------------------------------------------------------

candidate_file_no = 0
candidate_rows = 0


def write_candidates(rows):

    global candidate_file_no
    global candidate_rows

    if not rows:
        return

    # Convert directly from separate lists rather than a giant
    # object-array of Python tuples.
    s2_ids = [x[0] for x in rows]
    s3_ids = [x[1] for x in rows]

    df = pd.DataFrame({
        "source2_entity_id": pd.Series(
            s2_ids,
            dtype="string",
        ),
        "source3_entity_id": pd.Series(
            s3_ids,
            dtype="string",
        ),
    })

    # Small-batch deduplication only.
    df.drop_duplicates(
        inplace=True
    )

    if len(df) > 0:

        out = (
            CAND_DIR
            / f"candidate_{candidate_file_no:07d}.parquet"
        )

        df.to_parquet(
            out,
            index=False,
            compression="snappy",
        )

        candidate_rows += len(df)
        candidate_file_no += 1

    del df
    del s2_ids
    del s3_ids

    gc.collect()


# ---------------------------------------------------------------------
# STEP 2
# Process each Source 2 partition.
# ---------------------------------------------------------------------

print("\nGenerating candidate pairs...")

for s2_no, s2_path in enumerate(s2_parts):

    print(
        f"\nSource 2 partition "
        f"{s2_no + 1}/{len(s2_parts)}"
    )

    s2 = pd.read_parquet(s2_path)

    s2 = add_keys(s2)

    # Only keep what blocking needs.
    s2 = s2[
        [
            "entity_id",
            *KEY_COLUMNS,
        ]
    ]

    # Process ONE blocking key at a time.
    for key_no, key in enumerate(KEY_COLUMNS):

        print(
            f"  Blocking key "
            f"{key_no + 1}/{len(KEY_COLUMNS)}: "
            f"{key}"
        )

        # -------------------------------------------------------------
        # Build a compact map for this one Source 2 partition.
        # -------------------------------------------------------------

        s2_map = {}

        for value, group in s2.groupby(
            key,
            sort=False,
        ):

            if not value:
                continue

            n = len(group)

            if n > MAX_BLOCK_SIZE:
                continue

            s2_map[value] = (
                group["entity_id"]
                .astype(str)
                .tolist()
            )

        if not s2_map:
            continue

        # -------------------------------------------------------------
        # Scan Source 3 partitions.
        # -------------------------------------------------------------

        batch = []

        for s3_no, s3_path in enumerate(s3_index):

            s3 = pd.read_parquet(s3_path)

            # Only need this key.
            s3 = s3[
                [
                    "entity_id",
                    key,
                ]
            ]

            for value, group in s3.groupby(
                key,
                sort=False,
            ):

                if not value:
                    continue

                n = len(group)

                if n > MAX_BLOCK_SIZE:
                    continue

                s2_ids = s2_map.get(value)

                if not s2_ids:
                    continue

                s3_ids = (
                    group["entity_id"]
                    .astype(str)
                    .tolist()
                )

                # Cartesian product inside ONE small block.
                for s2_id in s2_ids:
                    for s3_id in s3_ids:

                        batch.append(
                            (
                                s2_id,
                                s3_id,
                            )
                        )

                        if len(batch) >= WRITE_BATCH_SIZE:

                            write_candidates(batch)

                            batch.clear()

            if batch:
                write_candidates(batch)
                batch.clear()

            del s3
            gc.collect()

        del s2_map
        gc.collect()

    del s2
    gc.collect()


# ---------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------

print("\n" + "=" * 80)
print("BLOCKING COMPLETE")
print("=" * 80)

print(
    f"Candidate files: "
    f"{candidate_file_no:,}"
)

print(
    f"Candidate rows written: "
    f"{candidate_rows:,}"
)

summary = pd.DataFrame([{
    "source2_partitions": len(s2_parts),
    "source3_partitions": len(s3_parts),
    "candidate_files": candidate_file_no,
    "candidate_rows": candidate_rows,
    "max_block_size": MAX_BLOCK_SIZE,
    "write_batch_size": WRITE_BATCH_SIZE,
}])

summary.to_csv(
    OUT / "blocking_summary.csv",
    index=False,
)

print("\nSaved:")
print("  outputs/blocking_summary.csv")
print("  outputs/candidate_pair_parts/")
