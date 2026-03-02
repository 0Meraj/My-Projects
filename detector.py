import joblib
import numpy as np
from features import build_manual_features

model = joblib.load("phishing_model.pkl")
vectorizer = joblib.load("tfidf_vectorizer.pkl")

def generate_explanation(url, pred):
    if pred == 1:
        return (
            "This URL contains patterns commonly associated with phishing attacks, "
            "such as brand impersonation, suspicious keywords, or abnormal structure."
        )
    else:
        return (
            "This URL does not contain strong phishing indicators and appears structurally normal."
        )

def predict_url(url):
    tfidf = vectorizer.transform([url]).toarray()
    manual = build_manual_features(url)

    features = np.hstack((tfidf, manual))

    pred = model.predict(features)[0]
    prob = model.predict_proba(features)[0][pred]

    return {
        "label": "PHISHING" if pred else "LEGITIMATE",
        "confidence": round(prob * 100, 2),
        "explanation": generate_explanation(url, pred)
    }

# Example
if __name__ == "__main__":
    test_url = "http://secure-paypa1.com/login"
    print(predict_url(test_url))