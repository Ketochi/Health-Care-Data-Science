import pandas as pd
import ast
import numpy as np
import pyarrow.parquet as pq

pd.set_option("display.max_columns", None)

# =====================================================
# LOAD ONLY NEEDED COLUMNS
# =====================================================

faers = pd.read_parquet(
    r"C:\Users\HP\OneDrive\Desktop\FAERS_processed\Final\faers_2.parquet",
    columns=[
        "drug_name",
        "drugindication",
        "activesubstance",
        "all_reactions",
        "RXCUI_list"
    ]
)

onside = pd.read_parquet(
    r"C:\Users\HP\OneDrive\Desktop\FAERS_processed\Final\onsides_2.parquet",
    columns=[
        "drug_name",
        "meddra_name",
        "RXCUI_list"
    ]
)

sider = pd.read_parquet(
    r"C:\Users\HP\OneDrive\Desktop\FAERS_processed\Final\sider_2.parquet",
    columns=[
        "Drug_name",
        "Indications_Combined",
        "Side_Effects_Combined",
        "RXCUI_list"
    ]
)


# =====================================================
# MEMORY OPTIMIZATION
# =====================================================

for df in [faers, onside, sider]:

    for col in df.select_dtypes(include="object").columns:

        df[col] = df[col].astype("string")

# =====================================================
# NORMALIZE LISTS
# =====================================================

def normalize_list(x):

    if isinstance(x, np.ndarray):
        return x.tolist()

    if isinstance(x, list):
        return x

    if x is None:
        return []

    if isinstance(x, float) and pd.isna(x):
        return []

    if isinstance(x, str):

        try:
            parsed = ast.literal_eval(x)

            if isinstance(parsed, list):
                return parsed

        except:
            return []

    return []

# =====================================================
# FLATTEN / CLEAN VALUES
# =====================================================

def flatten_values(values):

    cleaned = []

    for v in values.dropna():

        # numpy array
        if isinstance(v, np.ndarray):
            cleaned.extend(v.tolist())

        # python list
        elif isinstance(v, list):
            cleaned.extend(v)

        # semicolon-separated string
        elif isinstance(v, str):

            if ";" in v:
                cleaned.extend(
                    [i.strip() for i in v.split(";")]
                )

            else:
                cleaned.append(v)

        else:
            cleaned.append(v)

    # remove duplicates + empty strings
    cleaned = [
        str(x).strip()
        for x in cleaned
        if str(x).strip() != ""
    ]

    return sorted(list(set(cleaned)))

# =====================================================
# SAFE EXPLODE
# =====================================================

def explode_rx(df, col="RXCUI_list"):

    df[col] = df[col].apply(normalize_list)

    return df.explode(col)

# =====================================================
# EXPLODE LOOKUP TABLES
# =====================================================

onside_exp = explode_rx(onside)

sider_exp = explode_rx(sider)

# =====================================================
# AGGREGATE ONSIDES
# =====================================================

onside_lookup = (
    onside_exp
    .groupby("RXCUI_list")["meddra_name"]
    .agg(flatten_values)
    .reset_index()
)

# =====================================================
# AGGREGATE SIDER
# =====================================================

sider_lookup = (
    sider_exp
    .groupby("RXCUI_list")
    .agg({
        "Indications_Combined": flatten_values,
        "Side_Effects_Combined": flatten_values
    })
    .reset_index()
)

# free RAM
del onside_exp
del sider_exp

# =====================================================
# EXPLODE FAERS
# =====================================================

faers_exp = explode_rx(faers)

# =====================================================
# MERGE TYPES
# =====================================================

faers_exp["RXCUI_list"] = faers_exp["RXCUI_list"].astype("string")
onside_lookup["RXCUI_list"] = onside_lookup["RXCUI_list"].astype("string")
sider_lookup["RXCUI_list"] = sider_lookup["RXCUI_list"].astype("string")

# =====================================================
# MERGE
# =====================================================

merged = (
    faers_exp
    .merge(
        onside_lookup,
        on="RXCUI_list",
        how="left"
    )
    .merge(
        sider_lookup,
        on="RXCUI_list",
        how="left"
    )
)

# =====================================================
# COMBINE ALL SIDE EFFECTS
# =====================================================

merged["combined_side_effects"] = merged.apply(

    lambda row: flatten_values(
        pd.Series([

            row["all_reactions"],              # FAERS
            row["meddra_name"],                # ONSIDES
            row["Side_Effects_Combined"]       # SIDER

        ])
    ),

    axis=1
)

# =====================================================
# COMBINE ALL INDICATIONS
# =====================================================

merged["combined_indications"] = merged.apply(

    lambda row: flatten_values(
        pd.Series([

            row["drugindication"],             # FAERS
            row["Indications_Combined"]        # SIDER

        ])
    ),

    axis=1
)

# =====================================================
# FLAGS
# =====================================================

merged["in_onside"] = merged["meddra_name"].notna()

merged["in_sider"] = (
    merged["Indications_Combined"].notna()
)

# =====================================================
# FINAL OUTPUT
# =====================================================

final_dataset = merged[
    [
        "RXCUI_list",
        "drug_name",
        "activesubstance",

        # original columns
        "drugindication",
        "all_reactions",
        "meddra_name",
        "Indications_Combined",
        "Side_Effects_Combined",

        # unified columns
        "combined_indications",
        "combined_side_effects",

        "in_onside",
        "in_sider"
    ]
]

final_dataset.to_parquet(r"C:\Users\HP\OneDrive\Desktop\faers more\final.parquet")


print()
print("FAERS rows:", len(faers_exp))
print("ONSIDE grouped rows:", len(onside_lookup))
print("SIDER grouped rows:", len(sider_lookup))