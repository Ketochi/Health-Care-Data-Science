import pandas as pd
import re
import os
import pickle
from collections import defaultdict
from rapidfuzz import process
from tqdm import tqdm

pd.set_option("display.max_columns", None)
pd.set_option("display.max_colwidth", None)

# =========================
# LOAD DATA
# =========================

print("Loading ONSIDES...")

path = r"C:\Users\HP\OneDrive\Desktop\onsides-v3.1.0 (1)\csv\merged_onsides_translated.parquet"

df = pd.read_parquet(path)

# =========================
# CLEAN DRUG NAMES
# =========================

def clean_drug_name(name):

    if not isinstance(name, str):
        return ""

    s = name.lower()

    s = re.sub(r"\(.*?\)", "", s)
    s = re.sub(r"\[.*?\]", "", s)

    s = re.sub(
        r"\b\d+(\.\d+)?\s*(mg|mcg|ml|g|kg|iu|%)\b",
        "",
        s
    )

    s = re.sub(
        r"\b(tablet|tab|capsule|cap|injection|inj|solution|cream|gel|patch|spray|syrup|powder|liquid|xr|sr|er|cr|oral|turbuhaler)\b",
        "",
        s
    )

    s = re.sub(r"\b(and|with)\b", "", s)

    s = s.replace("/", " ")

    s = re.sub(r"[^a-z0-9 ]+", " ", s)

    return re.sub(r"\s+", " ", s).strip()

print("Cleaning drug names...")

df["drug_name"] = df["rxnorm_product_name"].apply(
    clean_drug_name
)

# =========================
# LOAD RXNORM
# =========================

print("Loading RxNorm...")

rxnconso_cols = [
    "RXCUI","LAT","TS","LUI","STT","SUI","ISPREF",
    "AUI","SAUI","SCUI","SDUI","SAB","TTY","CODE",
    "STR","SRL","SUPPRESS","CVF","EXTRA"
]

print("Reading RXNCONSO...")

rxnconso_full = pd.read_csv(
    r"C:\Users\HP\OneDrive\Desktop\RxNorm_full_10062025\rrf\RXNCONSO.RRF",
    sep="|",
    names=rxnconso_cols,
    dtype=str
).dropna(axis=1, how="all")

print("Filtering RxNorm...")

rxnconso = rxnconso_full[
    (rxnconso_full["SAB"] == "RXNORM") &
    (rxnconso_full["TTY"].isin([
        "IN","MIN","SY","BN","SCD","SBD","PIN"
    ]))
].copy()

print("Building ingredient set...")

ingredient_set = set(
    rxnconso.loc[
        rxnconso["TTY"] == "IN",
        "RXCUI"
    ]
)

print("Cleaning RxNorm strings...")

rxnconso["STR_clean"] = (
    rxnconso["STR"]
    .str.lower()
    .str.replace(r"[^a-z0-9 ]+", " ", regex=True)
    .str.replace(r"\s+", " ", regex=True)
    .str.strip()
)

print("Building lookup dictionary...")

rxnorm_lookup = defaultdict(list)

for _, row in tqdm(
    rxnconso.iterrows(),
    total=len(rxnconso),
    desc="RxNorm lookup"
):

    rxnorm_lookup[row["STR_clean"]].append(
        (row["RXCUI"], row["TTY"])
    )

rxnorm_strings = list(rxnorm_lookup.keys())

# =========================
# RXNREL MAPPING
# =========================

def build_rxnrel_map(path):

    cols = [
        "RXCUI1","AUI1","STYPE1","REL","RXCUI2","AUI2",
        "STYPE2","RELA","RUI","SL","SAB","RG","DIR",
        "SUPPRESS","EXTRA1","CVF","EXTRA2"
    ]

    print("Reading RXNREL...")

    df_rel = pd.read_csv(
        path,
        sep="|",
        names=cols,
        dtype=str
    ).dropna(axis=1, how="all")

    print("Filtering RXNREL...")

    df_rel = df_rel[df_rel["SAB"] == "RXNORM"]

    rel_map = defaultdict(list)

    print("Building relationship graph...")

    for _, r in tqdm(
        df_rel.iterrows(),
        total=len(df_rel),
        desc="RXNREL graph"
    ):

        if r["RELA"] in (
            "has_ingredient",
            "tradename_of",
            "consists_of"
        ):

            rel_map[r["RXCUI1"]].append(r["RXCUI2"])
            rel_map[r["RXCUI2"]].append(r["RXCUI1"])

    return rel_map

rel_map = build_rxnrel_map(
    r"C:\Users\HP\OneDrive\Desktop\RxNorm_full_10062025\rrf\RXNREL.RRF"
)

# =========================
# COLLAPSE TO INGREDIENT
# =========================

def collapse_to_ingredient(rxcui):

    visited = set()
    stack = [rxcui]

    while stack:

        cur = stack.pop()

        if cur in visited:
            continue

        visited.add(cur)

        stack.extend(rel_map.get(cur, []))

    ingredients = list(visited & ingredient_set)

    if ingredients:
        return ingredients

    return [rxcui]

# =========================
# MATCHING
# =========================

def match_drug_name(drug):

    cleaned = clean_drug_name(drug)

    # =====================
    # EXACT MATCH
    # =====================

    if cleaned in rxnorm_lookup:

        for rxcui, tty in rxnorm_lookup[cleaned]:

            ing = collapse_to_ingredient(rxcui)

            return {
                "rxcui": ing,
                "method": "exact"
            }

    # =====================
    # FUZZY MATCH
    # =====================

    if len(cleaned) >= 5:

        best = process.extractOne(
            cleaned,
            rxnorm_strings
        )

        if best and best[1] >= 85:

            for rxcui, tty in rxnorm_lookup[best[0]]:

                ing = collapse_to_ingredient(rxcui)

                return {
                    "rxcui": ing,
                    "method": "fuzzy"
                }

    return {
        "rxcui": [],
        "method": "unmatched"
    }

# =========================
# APPLY MATCHING
# =========================

cache_file = r"C:\Users\HP\OneDrive\Desktop\FAERS_processed\matches_onsides_cache.pkl"

if os.path.exists(cache_file):

    print("Loading matches from cache...")

    with open(cache_file, "rb") as f:
        matches = pickle.load(f)

else:

    print("Computing matches...")

    matches = {}

    unique_drugs = (
        df["drug_name"]
        .dropna()
        .unique()
    )

    for drug in tqdm(
        unique_drugs,
        desc="Matching"
    ):

        matches[drug] = match_drug_name(drug)

    with open(cache_file, "wb") as f:
        pickle.dump(matches, f)

# =========================
# APPLY TO DATAFRAME
# =========================

print("Applying matches...")

df["rxnorm_match"] = df["drug_name"].map(matches)

# =========================
# FINAL COLUMNS
# =========================

print("Creating columns...")

df["RXCUI_list"] = df["rxnorm_match"].apply(
    lambda x: x["rxcui"]
    if isinstance(x["rxcui"], list)
    else []
)

df["method"] = df["rxnorm_match"].apply(
    lambda x: x["method"]
)

# IMPORTANT:
# already ingredient-level
# NO SECOND EXPANSION
df["ingredients"] = df["RXCUI_list"]

# =========================
# SAVE
# =========================

print("Saving parquet...")

df.to_parquet(
    r"C:\Users\HP\OneDrive\Desktop\FAERS_processed\Final\onsides_2.parquet",
    index=False
)

print("DONE ✅")