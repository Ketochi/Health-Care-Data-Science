import duckdb, os

path_1 = r"C:\Users\HP\OneDrive\Desktop\FAERS_processed\merged_with_rxcui.parquet"
path_2 = r"C:\Users\HP\OneDrive\Desktop\FAERS_processed\onsides_with_RXCUI.parquet"
path_3 = r"C:\Users\HP\OneDrive\Desktop\FAERS_processed\sider_rxcui_single.parquet"

output_dir = r"C:\Users\HP\OneDrive\Desktop\FAERS_processed\compressed_chunks"
os.makedirs(output_dir, exist_ok=True)

con = duckdb.connect("pharma.db")

# Load parquet files
con.execute("CREATE OR REPLACE TABLE faers AS SELECT * FROM parquet_scan(?)", [path_1])
con.execute("CREATE OR REPLACE TABLE onsides AS SELECT * FROM parquet_scan(?)", [path_2])
con.execute("CREATE OR REPLACE TABLE sider AS SELECT * FROM parquet_scan(?)", [path_3])

# Flatten SIDER side effects
con.execute("""
CREATE OR REPLACE TABLE sider_flat AS
SELECT RXCUI, json_extract(se.value, '$.Side_Effect') AS Side_Effect
FROM sider, json_each("0") AS se;
""")

# Clean version of SIDER without column 0
con.execute("""
CREATE OR REPLACE TABLE sider_clean AS
SELECT 
    STITCH_ID,
    ATC,
    Drug_name,
    Indication_Name,
    drug_clean,
    RXCUI,
    TTY,
    method,
    ingredients,
    ingredient_name
FROM sider;
""")

# Get RXCUIs to process
rxcuis = con.execute("SELECT DISTINCT RXCUI FROM onsides").fetchall()

for (rxcui,) in rxcuis:
    out_file = os.path.join(output_dir, f"compressed_{rxcui}.parquet")
    con.execute(f"""
    COPY (
        SELECT 
            '{rxcui}' AS RXCUI,
            MAX(o.rxnorm_product_name) AS rxnorm_product_name,
            string_agg(DISTINCT f.drugindication, '; ') AS faers_indications,
            string_agg(DISTINCT o.meddra_name, '; ') AS onsides_indications,
            string_agg(DISTINCT si.indication, '; ') AS sider_indications,
            string_agg(DISTINCT sf.Side_Effect, '; ') AS all_side_effects
        FROM (SELECT * FROM onsides WHERE RXCUI = '{rxcui}') o
        LEFT JOIN (SELECT * FROM faers WHERE RXCUI = '{rxcui}') f ON o.RXCUI = f.RXCUI
        LEFT JOIN (SELECT * FROM sider_clean WHERE RXCUI = '{rxcui}') s ON o.RXCUI = s.RXCUI
        LEFT JOIN UNNEST(s.Indication_Name) AS si(indication) ON TRUE
        LEFT JOIN (SELECT * FROM sider_flat WHERE RXCUI = '{rxcui}') sf ON o.RXCUI = sf.RXCUI
    ) TO '{out_file}' (FORMAT PARQUET);
    """)
    print(f"Compressed RXCUI {rxcui} -> {out_file}")

# Merge all compressed chunks into one file
con.execute(f"""
COPY (
    SELECT * FROM parquet_scan('{output_dir}/compressed_*.parquet')
) TO '{output_dir}/final_compressed.parquet' (FORMAT PARQUET);
""")
