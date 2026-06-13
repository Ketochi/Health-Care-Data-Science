import pandas as pd
import numpy as np
import ast
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.linear_model import SGDClassifier
from sklearn.multiclass import OneVsRestClassifier
from sklearn.metrics import classification_report
from sklearn.decomposition import TruncatedSVD

from sentence_transformers import SentenceTransformer

# =====================================================
# LOAD DATA
# =====================================================

print("Loading data...")

df = pd.read_parquet(
    r"C:\Users\HP\OneDrive\Desktop\faers more\final.parquet"
)

# =====================================================
# HELPERS
# =====================================================

def safe_to_str(x):
    if x is None:
        return ""

    if isinstance(x, float) and pd.isna(x):
        return ""

    if isinstance(x, (list, np.ndarray)):
        return " ".join(map(str, x))

    return str(x)


def parse_side_effects(x):

    result = []

    if isinstance(x, (list, np.ndarray)):
        for item in x:
            result.extend(parse_side_effects(item))

    elif isinstance(x, str):

        try:
            parsed = ast.literal_eval(x)
            result.extend(parse_side_effects(parsed))

        except:
            result.append(x.strip())

    elif x is not None:
        result.append(str(x).strip())

    return result

# =====================================================
# TEXT FEATURES
# =====================================================

text_cols = [
    "drug_name",
    "activesubstance",
    "drugindication",
    "all_reactions",
    "combined_indications"
]

print("Creating text features...")

df["text_features"] = df[text_cols].apply(
    lambda row: " ".join(
        safe_to_str(v)
        for v in row
    ),
    axis=1
)

# =====================================================
# SIDE EFFECTS
# =====================================================

print("Parsing side effects...")

df["side_effects"] = (
    df["combined_side_effects"]
    .apply(parse_side_effects)
)

# =====================================================
# LABEL BINARIZATION
# =====================================================

print("Building labels...")

mlb = MultiLabelBinarizer(sparse_output=True)

Y = mlb.fit_transform(df["side_effects"])

label_counts = np.array(
    Y.sum(axis=0)
).ravel()

MIN_COUNT = 50

keep_mask = label_counts >= MIN_COUNT

Y = Y[:, keep_mask]
mlb.classes_ = mlb.classes_[keep_mask]

print("Labels kept:", len(mlb.classes_))

# =====================================================
# EMBEDDINGS
# =====================================================

embedder = SentenceTransformer(
    "all-MiniLM-L6-v2"
)

try:

    X = joblib.load(
        "cached_embeddings.pkl"
    )

    print("Loaded cached embeddings.")

except:

    print("Generating embeddings...")

    X = embedder.encode(
        df["text_features"].tolist(),
        show_progress_bar=True
    )

    joblib.dump(
        X,
        "cached_embeddings.pkl",
        compress=3
    )

# =====================================================
# SVD
# =====================================================

print("Applying SVD...")

svd = TruncatedSVD(
    n_components=100,
    random_state=42
)

X = svd.fit_transform(X)

# =====================================================
# TRAIN TEST SPLIT
# =====================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    Y,
    test_size=0.2,
    random_state=42
)

# =====================================================
# TRAIN MODEL
# =====================================================

print("Training model...")

clf = OneVsRestClassifier(
    SGDClassifier(
        loss="log_loss",
        max_iter=1000,
        random_state=42
    )
)

clf.fit(
    X_train,
    y_train
)

# =====================================================
# EVALUATION
# =====================================================

print("\nModel Evaluation:\n")

y_pred = clf.predict(X_test)

print(
    classification_report(
        y_test,
        y_pred,
        target_names=mlb.classes_
    )
)

# =====================================================
# DRUG EMBEDDING CACHE
# =====================================================

print("Building drug cache...")

drug_texts = (
    df.groupby("drug_name")["text_features"]
    .apply(" ".join)
    .to_dict()
)

drug_embeddings = {}

for drug, text in drug_texts.items():

    emb = embedder.encode([text])

    emb = svd.transform(emb)

    drug_embeddings[drug] = emb

# =====================================================
# SAVE EVERYTHING
# =====================================================

print("Saving files...")

joblib.dump(
    clf,
    "side_effect_model.pkl",
    compress=3
)

joblib.dump(
    mlb,
    "side_effect_labels.pkl",
    compress=3
)

joblib.dump(
    svd,
    "svd.pkl",
    compress=3
)

joblib.dump(
    drug_embeddings,
    "drug_embeddings.pkl",
    compress=3
)

joblib.dump(
    list(drug_embeddings.keys()),
    "drug_names.pkl",
    compress=3
)

print("\nDONE")
