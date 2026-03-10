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

    # Top 3 most significant flags for chat bubbles
    top_flags = [label for (_, label, _) in flagged[:3]]

    # All flags for technical report
    all_flags = [{"label": label, "severity": sev} for (_, label, sev) in flagged]

    result = {
        "label":      "PHISHING" if pred else "LEGITIMATE",
        "confidence": round(prob * 100, 2),
        "flags":      top_flags,
        "all_flags":  all_flags,
    }

    return result