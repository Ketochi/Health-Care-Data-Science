import pandas as pd
import numpy as np
import re
from collections import defaultdict
from rapidfuzz import process
from tqdm import tqdm

pd.set_option("display.max_columns", None)
pd.set_option("display.max_colwidth", None)

path1 = r"C:\Users\HP\OneDrive\Desktop\sider\drug_names.tsv"
path2 = r"C:\Users\HP\OneDrive\Desktop\sider\meddra_all_se.tsv"
path3 = r"C:\Users\HP\OneDrive\Desktop\sider\meddra_freq.tsv"
path4 = r"C:\Users\HP\OneDrive\Desktop\sider\meddra_all_indications.tsv"
path5 = r"C:\Users\HP\OneDrive\Desktop\sider\drug_atc.tsv"

# =========================================================
# PATHS
# =========================================================

SIDER_PATHS = {
    "drug_names": r"C:\Users\HP\OneDrive\Desktop\sider\drug_names.tsv",
    "atc": r"C:\Users\HP\OneDrive\Desktop\sider\drug_atc.tsv"
}

RXNORM_PATHS = {
    "conso": r"C:\Users\HP\OneDrive\Desktop\RxNorm_full_10062025\rrf\RXNCONSO.RRF",
    "rel": r"C:\Users\HP\OneDrive\Desktop\RxNorm_full_10062025\rrf\RXNREL.RRF"
}

OUTPUT_PATH = r"C:\Users\HP\OneDrive\Desktop\FAERS_processed\Final\sider_2.parquet"

# =========================================================
# CLEANING
# =========================================================

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
        r"\b(tablet|tab|capsule|cap|injection|inj|solution|cream|gel|patch|spray|syrup|powder|liquid|xr|sr|er|cr)\b",
        "",
        s
    )

    s = re.sub(r"\b(and|with)\b", "", s)

    s = re.sub(r"[^a-z0-9 ]+", " ", s)

    return re.sub(r"\s+", " ", s).strip()

# =========================================================
# LOAD SIDER
# =========================================================

# =========================================================
# LOAD SIDER
# =========================================================

def load_sider():

    print("Loading drug names...")
    drug_names = pd.read_csv(
        path1,
        sep="\t",
        names=["STITCH_ID", "Drug_name"]
    )

    # =====================================================
    # SIDE EFFECTS
    # =====================================================

    print("Loading side effects...")

    all_se = pd.read_csv(
        path2,
        sep="\t",
        names=[
            "STITCH_ID",
            "STITCH_FLAT_ID",
            "UMLS_ID",
            "MedDRA_Level",
            "MedDRA_ID",
            "Side_Effect"
        ]
    )

    all_se = all_se[
        all_se["MedDRA_Level"] == "PT"
    ]

    print("Combining side effects...")

    side_effects_grouped = (
        all_se.groupby("STITCH_ID")["Side_Effect"]
        .apply(
            lambda x: list(
                pd.Series(x).dropna().unique()
            )
        )
        .reset_index(name="Side_Effects_Combined")
    )

    # =====================================================
    # INDICATIONS
    # =====================================================

    print("Loading indications...")

    indications = pd.read_csv(
        path4,
        sep="\t",
        names=[
            "STITCH_ID",
            "UMLS_ID",
            "Evidence_Type",
            "Raw_Indication",
            "MedDRA_Level",
            "MedDRA_ID",
            "Indication_Name"
        ]
    )

    indications = indications[
        indications["MedDRA_Level"] == "PT"
    ]

    print("Combining indications...")

    indications_grouped = (
        indications.groupby("STITCH_ID")["Indication_Name"]
        .apply(
            lambda x: list(
                pd.Series(x).dropna().unique()
            )
        )
        .reset_index(name="Indications_Combined")
    )

    # =====================================================
    # ATC
    # =====================================================

    print("Loading ATC...")

    atc = pd.read_csv(
        path5,
        sep="\t",
        names=["STITCH_ID", "ATC"]
    )

    atc_grouped = (
        atc.groupby("STITCH_ID")["ATC"]
        .apply(
            lambda x: list(
                pd.Series(x).dropna().unique()
            )
        )
        .reset_index()
    )

    # =====================================================
    # FINAL MERGE
    # =====================================================

    print("Merging everything...")

    sider = (
        drug_names
        .merge(atc_grouped, on="STITCH_ID", how="left")
        .merge(indications_grouped, on="STITCH_ID", how="left")
        .merge(side_effects_grouped, on="STITCH_ID", how="left")
    )

    print("SIDER shape:", sider.shape)

    return sider
# =========================================================
# LOAD RXNORM
# =========================================================

def load_rxnorm():

    cols = [
        "RXCUI","LAT","TS","LUI","STT","SUI","ISPREF",
        "AUI","SAUI","SCUI","SDUI","SAB","TTY","CODE",
        "STR","SRL","SUPPRESS","CVF","EXTRA"
    ]

    print("Reading RXNCONSO...")
    rxnconso_full = pd.read_csv(
        RXNORM_PATHS["conso"],
        sep="|",
        names=cols,
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

    return rxnorm_lookup, rxnorm_strings, ingredient_set

# =========================================================
# BUILD RELATIONSHIPS
# =========================================================

def build_rxnrel_map(path):

    cols = [
        "RXCUI1","AUI1","STYPE1","REL","RXCUI2","AUI2",
        "STYPE2","RELA","RUI","SL","SAB","RG","DIR",
        "SUPPRESS","EXTRA1","CVF","EXTRA2"
    ]

    print("Reading RXNREL...")
    df = pd.read_csv(
        path,
        sep="|",
        names=cols,
        dtype=str
    )

    print("Filtering RXNREL...")
    df = df[df["SAB"] == "RXNORM"]

    rel_map = defaultdict(list)

    print("Building relationship graph...")
    for _, r in tqdm(
        df.iterrows(),
        total=len(df),
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

# =========================================================
# COLLAPSE TO INGREDIENT
# =========================================================

def collapse_to_ingredient(rxcui, rel_map, ingredient_set):

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

# =========================================================
# MATCHING
# =========================================================

def match_drug_name(
    drug,
    rxnorm_lookup,
    rxnorm_strings,
    rel_map,
    ingredient_set
):

    cleaned = clean_drug_name(drug)

    # =====================
    # EXACT
    # =====================

    if cleaned in rxnorm_lookup:

        for rxcui, tty in rxnorm_lookup[cleaned]:

            ing = collapse_to_ingredient(
                rxcui,
                rel_map,
                ingredient_set
            )

            return {
                "rxcui": ing,
                "method": "exact"
            }

    # =====================
    # FUZZY
    # =====================

    if len(cleaned) >= 5:

        best = process.extractOne(
            cleaned,
            rxnorm_strings
        )

        if best and best[1] >= 85:

            for rxcui, tty in rxnorm_lookup[best[0]]:

                ing = collapse_to_ingredient(
                    rxcui,
                    rel_map,
                    ingredient_set
                )

                return {
                    "rxcui": ing,
                    "method": "fuzzy"
                }

    return {
        "rxcui": [],
        "method": "unmatched"
    }

# =========================================================
# MAIN
# =========================================================

def main():

    # =====================
    # LOAD SIDER
    # =====================

    print("Loading SIDER...")
    sider = load_sider()

    print("Cleaning drug names...")
    sider["drug_clean"] = sider["Drug_name"].apply(
        clean_drug_name
    )

    # =====================
    # RXNORM
    # =====================

    print("Loading RxNorm...")
    rxnorm_lookup, rxnorm_strings, ingredient_set = load_rxnorm()

    # =====================
    # RELATIONSHIPS
    # =====================

    print("Building relationships...")
    rel_map = build_rxnrel_map(
        RXNORM_PATHS["rel"]
    )

    # =====================
    # MATCHING
    # =====================

    print("Matching drugs...")

    matches = {}

    unique_drugs = (
        sider["drug_clean"]
        .fillna("")
        .unique()
    )

    for drug in tqdm(
        unique_drugs,
        desc="Matching"
    ):

        matches[drug] = match_drug_name(
            drug,
            rxnorm_lookup,
            rxnorm_strings,
            rel_map,
            ingredient_set
        )

    # =====================
    # APPLY
    # =====================

    print("Applying matches...")

    sider["rxnorm_match"] = (
        sider["drug_clean"]
        .map(matches)
    )

    # =====================
    # FINAL COLUMNS
    # =====================

    print("Creating columns...")

    sider["RXCUI_list"] = sider["rxnorm_match"].apply(
        lambda x: x["rxcui"]
    )

    sider["method"] = sider["rxnorm_match"].apply(
        lambda x: x["method"]
    )

    # IMPORTANT:
    # already ingredient-level
    # NO SECOND EXPANSION
    sider["ingredients"] = sider["RXCUI_list"]

    # =====================
    # SAVE
    # =====================

    print("Saving parquet...")

    sider.to_parquet(
        OUTPUT_PATH,
        index=False
    )

    print("DONE ✅")

# =========================================================

if __name__ == "__main__":
    main()