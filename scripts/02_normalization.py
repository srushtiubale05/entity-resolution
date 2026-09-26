from pathlib import Path
import pandas as pd
import re
import unicodedata
import json

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
OUT = ROOT / "outputs"
OUT.mkdir(exist_ok=True)

# IMPORTANT:
# Do not create giant Python object columns (sets/lists) for 5M+ rows.
# Store compact normalized strings instead.

def normalize_basic(text):
    if pd.isna(text):
        return ""
    text = unicodedata.normalize("NFKC", str(text)).lower()
    chars = []
    for ch in text:
        cat = unicodedata.category(ch)
        chars.append(ch if cat.startswith(("L","N")) else " ")
    return re.sub(r"\s+", " ", "".join(chars)).strip()

def normalize_name(text):
    text = normalize_basic(text)
    for pattern, repl in [
        (r"\bprivate\s+limited\b", "pvt ltd"),
        (r"\bpvt\s+limited\b", "pvt ltd"),
        (r"\blimited\s+liability\s+company\b", "llc"),
    ]:
        text = re.sub(pattern, repl, text)
    mp = {
        "incorporated":"inc", "corporation":"corp", "company":"co",
        "limited":"ltd", "inc":"inc", "corp":"corp", "co":"co",
        "ltd":"ltd", "llc":"llc"
    }
    return " ".join(mp.get(t,t) for t in text.split())

ADDRESS_MAP = {
    "street":"st","st":"st","road":"rd","rd":"rd",
    "avenue":"ave","ave":"ave","boulevard":"blvd","blvd":"blvd",
    "drive":"dr","dr":"dr","lane":"ln","ln":"ln",
    "highway":"hwy","hwy":"hwy","parkway":"pkwy","pkwy":"pkwy",
    "place":"pl","pl":"pl","court":"ct","ct":"ct",
    "circle":"cir","cir":"cir","terrace":"ter","ter":"ter",
    "apartment":"apt","apt":"apt","suite":"ste","ste":"ste",
    "floor":"fl","fl":"fl","number":"no","no":"no"
}

def normalize_address(text):
    text = normalize_basic(text)
    return " ".join(ADDRESS_MAP.get(t,t) for t in text.split())

def process_file(src, dst, chunksize=100_000):
    first = True
    total = 0
    for df in pd.read_csv(
        src, sep="\t", dtype=str,
        usecols=["entity_id","business_name","business_address","country"],
        chunksize=chunksize
    ):
        df["business_name_raw"] = df["business_name"].fillna("")
        df["business_address_raw"] = df["business_address"].fillna("")
        df["business_name_norm"] = df["business_name_raw"].map(normalize_name)
        df["business_address_norm"] = df["business_address_raw"].map(normalize_address)
        df["country_norm"] = df["country"].fillna("").str.strip().str.lower()
        df.to_parquet(dst, engine="pyarrow", index=False, append=False) if False else None

        # Parquet append is awkward with plain pandas. Write partitioned CSV.
        part_dir = dst.parent / (dst.stem + "_parts")
        part_dir.mkdir(exist_ok=True)
        part_path = part_dir / f"part_{total//chunksize:05d}.parquet"
        df.to_parquet(part_path, index=False)
        total += len(df)

    return total

for n in ["source1","source2","source3"]:
    src = DATA / f"train_{n}.tsv"
    dst = OUT / f"{n}_norm.parquet"
    print(f"Normalizing {n}...")
    total = process_file(src, dst)
    print(f"  {total:,} rows")

print("\nNormalization complete.")
print("Normalized partition directories are in outputs/*_norm_parts/")
