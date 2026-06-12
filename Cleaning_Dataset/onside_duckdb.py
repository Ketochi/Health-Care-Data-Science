import duckdb
import os
import pandas as pd
import re
pd.set_option("Display.max_columns",None)
pd.set_option("Display.max_colwidth",None)

base = r"C:\Users\HP\OneDrive\Desktop\onsides-v3.1.0 (1)\csv"
con = duckdb.connect(database='onsides.db')

query = f"""
SET GLOBAL pandas_analyze_sample=0;

WITH effects AS (
    SELECT
        product_label_id,
        effect_meddra_id,
        label_section,
        match_method
    FROM read_csv('{base}/product_adverse_effect.csv', AUTO_DETECT=TRUE, ALL_VARCHAR=TRUE)
),

labels AS (
    SELECT
        label_id,
        source_product_name,
        source_label_url
    FROM read_csv('{base}/product_label.csv', AUTO_DETECT=TRUE, ALL_VARCHAR=TRUE)
),

label_to_rx AS (
    SELECT
        label_id,
        rxnorm_product_id
    FROM read_csv('{base}/product_to_rxnorm.csv', AUTO_DETECT=TRUE, ALL_VARCHAR=TRUE)
),

rxnorm_products AS (
    SELECT
        rxnorm_id AS product_id,
        rxnorm_name AS product_name
    FROM read_csv('{base}/vocab_rxnorm_product.csv', AUTO_DETECT=TRUE, ALL_VARCHAR=TRUE)
),

rxnorm_ing_to_prod AS (
    SELECT
        product_id,
        ingredient_id
    FROM read_csv('{base}/vocab_rxnorm_ingredient_to_product.csv', AUTO_DETECT=TRUE, ALL_VARCHAR=TRUE)
),

rxnorm_ingredients AS (
    SELECT
        rxnorm_id AS ingredient_id,
        rxnorm_name AS ingredient_name
    FROM read_csv('{base}/vocab_rxnorm_ingredient.csv', AUTO_DETECT=TRUE, ALL_VARCHAR=TRUE)
),

meddra AS (
    SELECT
        meddra_id,
        meddra_name,
        meddra_term_type
    FROM read_csv('{base}/vocab_meddra_adverse_effect.csv', AUTO_DETECT=TRUE, ALL_VARCHAR=TRUE)
),

highconf AS (
    SELECT
        ingredient_id,
        effect_meddra_id
    FROM read_csv('{base}/high_confidence.csv', AUTO_DETECT=TRUE, ALL_VARCHAR=TRUE)
)

SELECT
    e.product_label_id,
    l.source_product_name,
    l.source_label_url,
    lr.rxnorm_product_id,
    rp.product_name AS rxnorm_product_name,
    ri.ingredient_name,
    e.effect_meddra_id,              -- keep the raw MedDRA ID
    m.meddra_name,                   -- human-readable adverse effect
    m.meddra_term_type,              -- MedDRA hierarchy level
    e.label_section,
    e.match_method,
    CASE WHEN hc.ingredient_id IS NOT NULL THEN TRUE ELSE FALSE END AS high_confidence
FROM effects e
LEFT JOIN labels l
    ON e.product_label_id = l.label_id
LEFT JOIN label_to_rx lr
    ON l.label_id = lr.label_id
LEFT JOIN rxnorm_products rp
    ON lr.rxnorm_product_id = rp.product_id
LEFT JOIN rxnorm_ing_to_prod rip
    ON rp.product_id = rip.product_id
LEFT JOIN rxnorm_ingredients ri
    ON rip.ingredient_id = ri.ingredient_id
LEFT JOIN meddra m
    ON e.effect_meddra_id = m.meddra_id
LEFT JOIN highconf hc
    ON ri.ingredient_id = hc.ingredient_id
   AND e.effect_meddra_id = hc.effect_meddra_id;
"""

df = con.execute(query).fetchdf()
output_path = os.path.join(base, "merged_onsides2.parquet")
df.to_parquet(output_path, index=False)

print(f"✅ Saved merged dataset to: {output_path}")
print(df.head())
print(f"\n✅ Result shape: {df.shape}")

path = r"C:\Users\HP\OneDrive\Desktop\FAERS_processed\onsides_with_RXCUI.parquet"
df = pd.read_parquet(path)
df.drop(["rxnorm_match","TTY","method","high_confidence","source_product_name_en"],axis=1,inplace=True)

def clean_drug_name(name):
    if not isinstance(name, str):
        return ""

    s = name.lower()

    s = re.sub(r"\(.*?\)", "", s)
    s = re.sub(r"\[.*?\]", "", s)

    # remove strength
    s = re.sub(r"\b\d+(\.\d+)?\s*(mg|mcg|ml|g|kg|iu|%)\b", "", s)

    # remove standalone numbers
    s = re.sub(r"\b\d+\b", "", s)

    # remove dosage forms / pharma terms
    s = re.sub(
        r"\b(tablet|tab|capsule|cap|injection|inj|solution|cream|gel|patch|spray|syrup|powder|liquid|ophthalmic|topical|injectable|prefilled|syringe|release|extended|hr|xr|sr|er|cr)\b",
        "",
        s,
    )

    # remove salt forms
    s = re.sub(
        r"\b(hcl|hydrochloride|sodium|tartrate|sulfate|mesylate|hyclate|methylsulfate)\b",
        "",
        s,
    )

    s = re.sub(r"\b(and|with|for|per)\b", "", s)

    s = re.sub(r"[^a-z0-9 ]+", " ", s)

    return re.sub(r"\s+", " ", s).strip()

df["drug_clean"] = df["rxnorm_product_name"].apply(clean_drug_name)

df = (
    df.groupby("RXCUI_normalized")
    .agg({
        "rxnorm_product_name" : "first",
        "ingredient_name" : "first",
        "meddra_name" : lambda x: list(set(x)),
        "drug_clean" : "first"
    })
)
print(df[["rxnorm_product_name","drug_clean"]].sample(frac=0.002))

