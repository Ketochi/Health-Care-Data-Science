import joblib
import numpy as np

from difflib import get_close_matches

# =====================================================
# LOAD SAVED FILES
# =====================================================

print("Loading model...")

clf = joblib.load(
    "side_effect_model.pkl"
)

mlb = joblib.load(
    "side_effect_labels.pkl"
)

drug_embeddings = joblib.load(
    "drug_embeddings.pkl"
)

drug_names = joblib.load(
    "drug_names.pkl"
)

print("Ready.\n")

# =====================================================
# MEMORY CACHE
# =====================================================

prediction_cache = {}

# =====================================================
# DRUG MATCHING
# =====================================================

def find_best_match(drug_name):

    match = get_close_matches(
        drug_name,
        drug_names,
        n=1,
        cutoff=0.6
    )

    if match:
        return match[0]

    return None

# =====================================================
# PREDICTION
# =====================================================

def predict_side_effects(
    drug_name,
    top_k=15,
    min_probability=0.10
):

    drug_name = drug_name.strip()

    if drug_name in prediction_cache:
        return prediction_cache[drug_name]

    best_match = find_best_match(
        drug_name
    )

    if best_match is None:

        return {
            "error":
            f"No matching drug found for '{drug_name}'"
        }

    emb = drug_embeddings[best_match]

    probs = clf.predict_proba(
        emb
    )[0]

    ranked_idx = np.argsort(
        probs
    )[::-1]

    results = []

    for idx in ranked_idx:

        prob = float(
            probs[idx]
        )

        if prob < min_probability:
            continue

        results.append({
            "side_effect":
                mlb.classes_[idx],
            "probability":
                round(prob, 4)
        })

        if len(results) >= top_k:
            break

    output = {
        "drug": best_match,
        "predicted_side_effects": results
    }

    prediction_cache[
        drug_name
    ] = output

    return output

# =====================================================
# INTERACTIVE MODE
# =====================================================

while True:

    drug = input(
        "\nDrug Name (or 'quit'): "
    )

    if drug.lower() == "quit":
        break

    result = predict_side_effects(
        drug
    )

    if "error" in result:

        print(
            "\n",
            result["error"]
        )

        continue

    print(
        f"\nMatched Drug: {result['drug']}"
    )

    print(
        "\nPredicted Side Effects:\n"
    )

    for item in result[
        "predicted_side_effects"
    ]:

        print(
            f"{item['side_effect']} "
            f"({item['probability']:.4f})"
        )