import os
import json
import pandas as pd
from pandas import json_normalize
from tqdm import tqdm

# ====== CONFIG ======
INPUT_FOLDER = r"C:\Users\HP\OneDrive\Desktop\faers more"
OUTPUT_FOLDER = r"C:\Users\HP\OneDrive\Desktop\FAERS_processed"
os.makedirs(OUTPUT_FOLDER, exist_ok=True)
pd.set_option("display.max_columns", None)
# ====== FDA ROUTE CODE MAPPING ======
fda_route_codes = {
    "ORAL": ["001"],
    "INTRAVENOUS": ["002", "138", "137"],
    "PARENTERAL": ["065", "064"],
    "SUBCUTANEOUS": ["003", "058", "58"],
    "INTRAMUSCULAR": ["005"],
    "TOPICAL": ["011"],
    "RESPIRATORY (INHALATION)": ["136"],
    "INTRAPERITONEAL": ["004", "033", "33"],
    "TRANSDERMAL": ["041", "062", "358"],
    "RECTAL": ["016"],
    "VAGINAL": ["015"],
    "BUCCAL": ["030"],
    "NASAL": ["014"],
    "OPHTHALMIC": ["012"],
    "AURICULAR (OTIC)": ["013"],
    "UNKNOWN": ["139"],
}
route_lookup = {code: route for route, codes in fda_route_codes.items() for code in codes}


# ====== HELPER FUNCTION ======
def process_faers_json(file_path):
    """Process one FAERS JSON file and return a cleaned DataFrame."""
    with open(file_path, "r") as f:
        data = json.load(f)

    results = data.get("results", [])
    if not results:
        print(f"⚠️ Skipped empty file: {os.path.basename(file_path)}")
        return pd.DataFrame()

    df = json_normalize(results)

    # ---------------------------
    # 🧩 SERIOUSNESS FLAGS
    # ---------------------------
    # ---------------------------
    # 🧩 SERIOUSNESS FLAGS
    # ---------------------------
    # ---------------------------
    # 🧩 SERIOUSNESS FLAGS
    # ---------------------------
    seriousness_cols = [c for c in df.columns if c.startswith("seriousness") and c != "serious"]

    # Normalize seriousness flags: 1 = serious, everything else = not serious
    for col in seriousness_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)
        df[col] = df[col].apply(lambda x: 1 if x == 1 else 0)

    # Binary serious flag: 1 if any seriousness flag is 1, else 0
    df["serious"] = df[seriousness_cols].max(axis=1) if seriousness_cols else 0

    # ---------------------------
    # 👤 PATIENT INFO
    # ---------------------------
    for col, label in {
        "patient.patientonsetage": "age",
        "patient.patientweight": "weight"
    }.items():
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
            median_val = df[col].median()
            df[col] = df[col].fillna(median_val)
        else:
            df[col] = None
    # Patient gender
    if "patient.patientsex" in df.columns:
        # FAERS codes: 1 = male, 2 = female, 0 or missing = unknown
        gender_map = {1: "male", 2: "female"}
        df["gender"] = pd.to_numeric(df["patient.patientsex"], errors="coerce").map(gender_map).fillna("unknown")
    else:
        df["gender"] = "unknown"


    # ---------------------------
    # 💊 DRUG & REACTION EXTRACTION
    # ---------------------------
    def extract_field(drug_list, key):
        """Extracts the first valid string from list of dicts under a given key."""
        if not isinstance(drug_list, list):
            return None
        for d in drug_list:
            if isinstance(d, dict) and key in d and d[key]:
                return d[key]
        return None

    # Drug name, indication, and active substance
    df["drug_name"] = df["patient.drug"].apply(lambda x: extract_field(x, "medicinalproduct"))
    df["drugindication"] = df["patient.drug"].apply(lambda x: extract_field(x, "drugindication"))
    df["activesubstance"] = df["patient.drug"].apply(
        lambda x: extract_field(x, "activesubstance.activesubstancename")
    )

    # Route mapping
    def map_route(drug_list):
        if not isinstance(drug_list, list) or len(drug_list) == 0:
            return "UNKNOWN"
        code = str(drug_list[0].get("drugadministrationroute", "139")).zfill(3)
        return route_lookup.get(code, "UNKNOWN")

    df["route"] = df["patient.drug"].apply(map_route)

    # ---------------------------
    # 💬 REACTIONS (semicolon-separated)
    # ---------------------------
    def extract_reactions(reaction_list, key):
        """Extracts all unique values for a given key from patient.reaction list."""
        if not isinstance(reaction_list, list):
            return ""
        return "; ".join(sorted({str(d.get(key)) for d in reaction_list if key in d and d[key]}))

    # Reaction PT text (MedDRA Preferred Term)
    df["all_reactions"] = df["patient.reaction"].apply(
        lambda x: extract_reactions(x, "reactionmeddrapt")
    )

    # Reaction MedDRA version (dictionary version used)
    df["reaction_meddra_version"] = df["patient.reaction"].apply(
        lambda x: extract_reactions(x, "reactionmeddraversionpt")
    )

    # ---------------------------
    # 🧹 FINAL SELECTION
    # ---------------------------
    keep_cols = [
        "safetyreportid",
        "receiptdate",
        "drug_name",
        "drugindication",
        "activesubstance",
        "route",
        "all_reactions",
        "reaction_meddra_version",  # <-- NEW
        "serious",
        "patient.patientonsetage",
        "patient.patientweight",
        "gender"
    ]

    df = df[keep_cols]

    return df


# ====== MAIN LOOP ======
for file in tqdm(os.listdir(INPUT_FOLDER), desc="Processing FAERS files"):
    if not file.endswith(".json"):
        continue

    file_path = os.path.join(INPUT_FOLDER, file)
    cleaned_df = process_faers_json(file_path)

    if cleaned_df.empty:
        continue

    output_path = os.path.join(
        OUTPUT_FOLDER, os.path.splitext(file)[0] + "_cleaned.parquet"
    )
    cleaned_df.to_parquet(output_path, index=False)
    print(f"✅ Saved cleaned file: {output_path}")

print("🏁 All FAERS files processed successfully!")


