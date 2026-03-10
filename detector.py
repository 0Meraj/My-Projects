import joblib
import numpy as np
from features import build_manual_features

model = joblib.load("phishing_model.pkl")
vectorizer = joblib.load("tfidf_vectorizer.pkl")


def predict_url(url):
    tfidf = vectorizer.transform([url]).toarray()
    features_array, flagged = build_manual_features(url)

    features = np.hstack((tfidf, features_array))

    pred = model.predict(features)[0]
    prob = model.predict_proba(features)[0][pred]

    # Top 3 most significant flags (human labels only)
    top_flags = [label for (_, label, _) in flagged[:3]]

    result = {
        "label":      "PHISHING" if pred else "LEGITIMATE",
        "confidence": round(prob * 100, 2),
        "flags":      top_flags,
    }

    return result