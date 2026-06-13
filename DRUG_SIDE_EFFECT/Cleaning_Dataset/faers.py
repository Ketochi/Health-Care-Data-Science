import pandas as pd
import re
import os
import pickle
from collections import defaultdict
from rapidfuzz import process
from tqdm import tqdm
from functools import lru_cache

pd.set_option("display.max_columns", None)
pd.set_option("display.max_colwidth", None)

# =========================
# Load FAERS
# =========================
path1 = r"C:\Users\HP\OneDrive\Desktop\FAERS_processed\merged.parquet"
faers_drugs = pd.read_parquet(path1)

# =========================
# Clean drug names
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
        r"\b(tablet|tab|capsule|cap|injection|inj|solution|cream|gel|patch|spray|syrup|powder|liquid|xr|sr|er|cr|extra strength|migraine|jr|pf|nos)\b",
        "",
        s
    )

    s = re.sub(r"\b(and|with)\b", "", s)

    s = re.sub(r"[^a-z0-9 ]+", " ", s)

    return re.sub(r"\s+", " ", s).strip()

# =========================
# Manual mapping
# =========================
manual_map = {
    "cipralex": "escitalopram",
    "glivec": "imatinib",
    "contoloc": "pantoprazole",
    "excedrin": "acetaminophen aspirin caffeine",
    "sensodyne": "sodium fluoride",
    "theraflu": "acetaminophen diphenhydramine phenylephrine",
}

def apply_manual_map(name):
    cleaned = clean_drug_name(name)
    return manual_map.get(cleaned, cleaned)

faers_drugs["drug_clean"] = faers_drugs["drug_name"].apply(apply_manual_map)

# =========================
# Load RxNorm
# =========================
rxnconso_cols = [
    "RXCUI","LAT","TS","LUI","STT","SUI","ISPREF","AUI",
    "SAUI","SCUI","SDUI","SAB","TTY","CODE","STR",
    "SRL","SUPPRESS","CVF","EXTRA"
]

print("Loading RXNCONSO...")

rxnconso_full = pd.read_csv(
    r"C:\Users\HP\OneDrive\Desktop\RxNorm_full_10062025\rrf\RXNCONSO.RRF",
    sep="|",
    names=rxnconso_cols,
    dtype=str,
    low_memory=False
).dropna(axis=1, how="all")

rxnconso = rxnconso_full[
    (rxnconso_full["SAB"] == "RXNORM") &
    (rxnconso_full["TTY"].isin(["IN","MIN","SY","BN","SCD","SBD","PIN"]))
].copy()

ingredient_set = set(
    rxnconso.loc[rxnconso["TTY"] == "IN", "RXCUI"]
)

print("Cleaning RxNorm strings...")

rxnconso["STR_clean"] = (
    rxnconso["STR"]
    .str.lower()
    .str.replace(r"[^a-z0-9 ]+", " ", regex=True)
    .str.replace(r"\s+", " ", regex=True)
    .str.strip()
)

# =========================
# FAST LOOKUP TABLE
# =========================
print("Building lookup tables...")

rxnorm_lookup = defaultdict(list)

for row in rxnconso[
    ["STR_clean", "RXCUI", "TTY"]
].itertuples(index=False):

    rxnorm_lookup[row.STR_clean].append(
        (row.RXCUI, row.TTY)
    )

rxnorm_strings = list(rxnorm_lookup.keys())

# =========================
# RXNREL mapping
# =========================
def build_rxnrel_map(path):

    cols = [
        "RXCUI1","AUI1","STYPE1","REL","RXCUI2","AUI2",
        "STYPE2","RELA","RUI","SL","SAB","RG","DIR",
        "SUPPRESS","EXTRA1","CVF","EXTRA2"
    ]

    print("Loading RXNREL...")

    df_rel = pd.read_csv(
        path,
        sep="|",
        names=cols,
        dtype=str,
        low_memory=False
    )

    df_rel = df_rel[
        (df_rel["SAB"] == "RXNORM") &
        (
            df_rel["RELA"].isin([
                "has_ingredient",
                "tradename_of",
                "consists_of"
            ])
        )
    ]

    rel_map = defaultdict(list)

    print("Building relationship graph...")

    for row in df_rel[
        ["RXCUI1", "RXCUI2"]
    ].itertuples(index=False):

        rel_map[row.RXCUI1].append(row.RXCUI2)
        rel_map[row.RXCUI2].append(row.RXCUI1)

    return rel_map

rel_map = build_rxnrel_map(
    r"C:\Users\HP\OneDrive\Desktop\RxNorm_full_10062025\rrf\RXNREL.RRF"
)

# =========================
# FAST CACHED INGREDIENT COLLAPSE
# =========================
@lru_cache(maxsize=None)
def collapse_to_ingredient_cached(rxcui):

    visited = set()
    stack = [rxcui]

    while stack:
        cur = stack.pop()

        if cur in visited:
            continue

        visited.add(cur)

        stack.extend(rel_map.get(cur, []))

    return tuple(visited & ingredient_set)

# =========================
# MATCHING
# =========================
def match_drug_name(drug):

    cleaned = clean_drug_name(drug)

    # -------- Exact --------
    if cleaned in rxnorm_lookup:

        for rxcui, tty in rxnorm_lookup[cleaned]:

            ing = list(
                collapse_to_ingredient_cached(rxcui)
            )

            if ing:
                return {
                    "rxcui": ing,
                    "method": "exact"
                }

            return {
                "rxcui": [rxcui],
                "method": "exact"
            }

    # -------- Fuzzy --------
    if len(cleaned) >= 5:

        best = process.extractOne(
            cleaned,
            rxnorm_strings
        )

        if best and best[1] >= 85:

            for rxcui, tty in rxnorm_lookup[best[0]]:

                ing = list(
                    collapse_to_ingredient_cached(rxcui)
                )

                if ing:
                    return {
                        "rxcui": ing,
                        "method": "fuzzy"
                    }

                return {
                    "rxcui": [rxcui],
                    "method": "fuzzy"
                }

    return {
        "rxcui": [],
        "method": "unmatched"
    }

# =========================
# APPLY MATCHING
# =========================
cache_file = r"C:\Users\HP\OneDrive\Desktop\FAERS_processed\matches_cache.pkl"

if os.path.exists(cache_file):

    print("Loading matches from cache...")

    with open(cache_file, "rb") as f:
        matches = pickle.load(f)

else:

    print("Computing matches...")

    matches = {}

    unique_drugs = (
        faers_drugs["drug_clean"]
        .fillna("")
        .unique()
    )

    for drug in tqdm(unique_drugs, desc="Matching"):

        matches[drug] = match_drug_name(drug)

    with open(cache_file, "wb") as f:
        pickle.dump(matches, f)

# =========================
# MAP RESULTS
# =========================
print("Applying matches...")

faers_drugs["rxnorm_match"] = (
    faers_drugs["drug_clean"]
    .map(matches)
)

def extract_rxcui(x):
    if isinstance(x, dict):
        val = x.get("rxcui", [])
        return val if isinstance(val, list) else []
    return []

def extract_method(x):
    if isinstance(x, dict):
        return x.get("method", "unmatched")
    return "unmatched"

faers_drugs["RXCUI_list"] = (
    faers_drugs["rxnorm_match"]
    .apply(extract_rxcui)
)

faers_drugs["method"] = (
    faers_drugs["rxnorm_match"]
    .apply(extract_method)
)
# =========================
# INGREDIENT COLUMN
# =========================
# IMPORTANT:
# already ingredient-level from matcher
# so NO expensive expansion step needed

faers_drugs["ingredients"] = (
    faers_drugs["RXCUI_list"]
)

# =========================
# SAVE
# =========================
print("Saving parquet...")

faers_drugs.to_parquet(
    r"C:\Users\HP\OneDrive\Desktop\FAERS_processed\Final\faers_2.parquet",
    index=False
)

print("DONE ✅")
print(faers_drugs[[
    "drug_name",
    "drug_clean",
    "RXCUI_list",
    "ingredients",
    "method"
]].head())
