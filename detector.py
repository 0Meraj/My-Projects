import joblib
import re
import numpy as np
import unicodedata
from urllib.parse import urlparse
from rapidfuzz.distance import Levenshtein
import whois
from datetime import datetime
import math

model = joblib.load("phishing_model.pkl")
vectorizer = joblib.load("tfidf_vectorizer.pkl")

protected_brands = [
    "paypal", "google", "facebook", "microsoft", "apple",
    "linkedin", "twitter", "amazon", "github", "bankofamerica", "instagram"
]

suspicious_keywords = {
    "login", "verify", "update", "secure",
    "account", "signin", "password"
}

suspicious_paths = {
    "login", "verify", "secure", "account", "update",
    "signin", "auth", "password", "confirm"
}

homoglyph_map = str.maketrans({
    '0': 'o', '1': 'l', '3': 'e', '5': 's',
    '7': 't', '@': 'a', '$': 's', '!': 'i'
})

def extract_domain(url):
    return re.sub(r'^(www\.|m\.)', '', urlparse(url).netloc.lower())

def extract_path(url):
    return urlparse(url).path.lower()

def normalize_domain(domain):
    domain = unicodedata.normalize('NFKC', domain).translate(homoglyph_map)
    tokens = re.split(r'[.\-_/]+', domain)
    return [t for t in tokens if t and not t.isdigit()]

def is_punycode(domain):
    return any(label.startswith("xn--") for label in domain.split("."))

def domain_brand_distance(tokens):
    min_dist = float("inf")
    suspicious = 0
    for token in tokens:
        for brand in protected_brands:
            d = Levenshtein.distance(token, brand)
            min_dist = min(min_dist, d)
            if d <= 1 and token != brand:
                suspicious += 1
        if token in suspicious_keywords:
            suspicious += 1
    return min_dist, suspicious

def shannon_entropy(s):
    if not s:
        return 0.0
    probs = [s.count(c) / len(s) for c in set(s)]
    return -sum(p * math.log2(p) for p in probs)

def path_suspicion_score(url):
    tokens = re.split(r"[\/\-_]", extract_path(url))
    hits = sum(1 for t in tokens if t in suspicious_paths)
    return hits, int(hits > 0)

def extract_url_heuristics(url):
    domain = extract_domain(url)
    try:
        w = whois.whois(domain)
        cd = w.creation_date
        if isinstance(cd, list):
            cd = cd[0]
        age = (datetime.utcnow() - cd).days if cd else -1
    except Exception:
        age = -1

    return {
        "URL Length": len(url),
        "Subdomain Count": max(len(domain.split(".")) - 2, 0),
        "Hyphen Count": url.count("-"),
        "Contains IP": int(bool(re.match(r"^\d{1,3}(\.\d{1,3}){3}$", domain))),
        "Contains @": int("@" in url),
        "Domain Age (days)": age,
        "New Domain (<30d)": int(0 <= age < 30),
        "Very New Domain (<7d)": int(0 <= age < 7)
    }

def generate_explanation(url, is_phishing):
    domain = extract_domain(url)
    tokens = normalize_domain(domain)

    min_lev, suspicious_count = domain_brand_distance(tokens)
    entropy = shannon_entropy(domain)
    path_hits, has_suspicious_path = path_suspicion_score(url)
    h = extract_url_heuristics(url)

    lines = []

    if is_phishing:
        lines.append(
            "This link shows several warning signs that are commonly seen in scam websites."
        )

        if min_lev <= 1:
            lines.append(
                "The website name closely imitates a well-known brand with small changes, a common phishing technique."
            )

        if suspicious_count > 0:
            lines.append(
                "The domain contains terms designed to create urgency or pressure."
            )

        if has_suspicious_path:
            lines.append(
                "The page path focuses on account access or verification, which is frequently abused by phishing sites."
            )

        if h["Very New Domain (<7d)"]:
            lines.append(
                "The domain was registered very recently, which is typical of short-lived scam campaigns."
            )

        if h["Hyphen Count"] > 1:
            lines.append(
                "Multiple hyphens are often used to mimic legitimate domains."
            )

        if entropy > 4.0:
            lines.append(
                "The domain name appears unusually random compared to established services."
            )

        lines.append(
            "It would be safer not to enter sensitive information on this page."
        )

    else:
        lines.append(
            "This link does not show the common warning signs typically associated with phishing websites."
        )

        if min_lev > 1:
            lines.append(
                "The domain name matches the organisation it claims to represent without suspicious variations."
            )

        if suspicious_count == 0:
            lines.append(
                "The domain avoids language intended to rush or alarm users."
            )

        if not has_suspicious_path:
            lines.append(
                "The page structure appears consistent with legitimate websites."
            )

        if h["Domain Age (days)"] > 180:
            lines.append(
                "The domain has existed for a long time, which aligns with legitimate services."
            )

        if entropy < 4.0:
            lines.append(
                "The domain name is clear and readable rather than random."
            )

        lines.append(
            "While no automated system can guarantee safety, this link behaves consistently with trusted websites."
        )

    return " ".join(lines)

def predict_url(url):
    domain = extract_domain(url)
    tokens = normalize_domain(domain)

    min_lev, suspicious_count = domain_brand_distance(tokens)
    entropy = shannon_entropy(domain)
    path_hits, has_suspicious_path = path_suspicion_score(url)
    heuristics = extract_url_heuristics(url)

    tfidf = vectorizer.transform([url]).toarray()
    manual = np.array([[
        min_lev,
        suspicious_count,
        (suspicious_count > 0) * 2 + (min_lev <= 1),
        entropy,
        int(is_punycode(domain)),
        heuristics["URL Length"],
        heuristics["Subdomain Count"],
        heuristics["Hyphen Count"],
        heuristics["Contains IP"],
        heuristics["Contains @"],
        heuristics["Domain Age (days)"],
        heuristics["New Domain (<30d)"],
        heuristics["Very New Domain (<7d)"],
        path_hits,
        has_suspicious_path
    ]])

    features = np.hstack((tfidf, manual))
    pred = model.predict(features)[0]
    prob = model.predict_proba(features)[0][pred]

    system_report = {
        "Prediction": "PHISHING" if pred else "LEGITIMATE",
        "Confidence (%)": round(prob * 100, 2),
        "Min Brand Distance": min_lev,
        "Suspicious Token Count": suspicious_count,
        "Domain Entropy": round(entropy, 3),
        "Suspicious Path Hits": path_hits,
        "Has Suspicious Path": bool(has_suspicious_path),
        "Internationalized Domain": is_punycode(domain),
        **heuristics
    }

    return {
        "label": system_report["Prediction"],
        "confidence": system_report["Confidence (%)"],
        "explanation": generate_explanation(url, bool(pred)),
        "system_report": system_report
    }
