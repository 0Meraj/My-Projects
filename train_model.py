import pandas as pd
import numpy as np
import joblib
import re
import json
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report

from features import build_manual_features, MANUAL_FEATURE_ORDER

# Load dataset
df = pd.read_parquet(r"C:\Users\meroq\Downloads\archive\Training.parquet")

# Extract URLs and labels
phishing_urls = df[df["status"] == "phishing"]["url"].dropna().astype(str).tolist()
legitimate_urls = df[df["status"] == "legitimate"]["url"].dropna().astype(str).tolist()

# Combine
data = pd.DataFrame({
    "url": phishing_urls + legitimate_urls,
    "label": [1]*len(phishing_urls) + [0]*len(legitimate_urls)
})

# Shuffle dataset
data = data.sample(frac=1, random_state=42).reset_index(drop=True)

# ---------------------------
# TF-IDF
# ---------------------------

vectorizer = TfidfVectorizer(
    analyzer="char",
    ngram_range=(3, 5),
    max_features=5000
)

X_tfidf = vectorizer.fit_transform(data["url"]).toarray()

# ---------------------------
# Manual features
# ---------------------------

manual_features = np.vstack(
    data["url"].apply(build_manual_features)
)

X = np.hstack((X_tfidf, manual_features))
y = data["label"]

# ---------------------------
# Train
# ---------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.1, random_state=42, stratify=y
)

model = RandomForestClassifier(
    n_estimators=200,
    random_state=42,
    n_jobs=-1,
    class_weight={0: 1, 1: 2}
)

model.fit(X_train, y_train)

report = classification_report(y_test, model.predict(X_test), zero_division=0, output_dict=True)
print(classification_report(y_test, model.predict(X_test), zero_division=0))

with open("report.json", "w") as f:
    json.dump(report, f)

print("report.json saved.")

# ---------------------------
# Save
# ---------------------------

joblib.dump(model, "phishing_model.pkl")
joblib.dump(vectorizer, "tfidf_vectorizer.pkl")
joblib.dump(MANUAL_FEATURE_ORDER, "feature_order.pkl")