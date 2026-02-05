import pandas as pd
import re
import numpy as np
import unicodedata
from urllib.parse import urlparse
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from rapidfuzz.distance import Levenshtein
import joblib
import whois
from datetime import datetime, timezone
import math

phishing_urls = [
    "http://paypal-secure-login.com/verify",
    "https://secure-paypa1.com/login",
    "http://account-google.verify-user.net",
    "https://facebook-login-security.com/auth",
    "http://appleid-confirmation.net/login",
    "https://microsoft-account-secure.com/update",
    "http://amazon-verification-alert.com/signin",
    "https://bankofamerica-secure-check.com/login",
    "http://instagram-security-alert.net/verify",
    "https://github-account-authenticate.com/login",
    "http://paypal.com.user-confirmation.ru/login",
    "https://secure-google-account-update.info",
    "http://login-facebook-helpdesk.com/auth",
    "https://apple-support-verifyid.net/confirm",
    "http://amazon-account-protection.co/verify"
]

legitimate_urls = [
    "https://www.google.com",
    "https://www.facebook.com",
    "https://www.microsoft.com",
    "https://www.apple.com",
    "https://www.amazon.com",
    "https://www.paypal.com",
    "https://www.github.com",
    "https://www.linkedin.com",
    "https://www.twitter.com",
    "https://www.instagram.com",
    "https://www.bankofamerica.com",
    "https://www.wikipedia.org",
    "https://www.bbc.co.uk",
    "https://www.stackoverflow.com",
    "https://www.reddit.com"
]

phish = pd.DataFrame({
    "url": phishing_urls,
    "label": 1
})

legit = pd.DataFrame({
    "url": legitimate_urls,
    "label": 0
})

print("PHISH DATASET:", phish.shape)
print("LEGIT DATASET:", legit.shape)

data = pd.concat([phish, legit], ignore_index=True)
print("TOTAL DATASET:", data.shape)

def extract_domain(url):
    parsed = urlparse(url)
    return re.sub(r'^(www\.|m\.)', '', parsed.netloc.lower())

phish_domains = set(phish["url"].apply(extract_domain))
legit = legit[~legit["url"].apply(extract_domain).isin(phish_domains)]

data = pd.concat([phish, legit], ignore_index=True)
print("TOTAL DATASET:", data.shape)

protected_brands = [
    "paypal", "google", "facebook", "microsoft", "apple",
    "linkedin", "twitter", "amazon", "github",
    "bankofamerica", "instagram"
]

suspicious_keywords = {
    "login", "verify", "update", "secure",
    "account", "signin", "password"
}

suspicious_paths = {
    "login", "verify", "secure", "account",
    "update", "signin", "auth", "password", "confirm"
}

homoglyph_map = str.maketrans({
    '0': 'o', '1': 'l', '3': 'e', '5': 's',
    '7': 't', '@': 'a', '$': 's', '€': 'e', '!': 'i'
})

def extract_path(url):
    return urlparse(url).path.lower()

def is_punycode(domain):
    return any(label.startswith("xn--") for label in domain.split("."))

def normalize_domain(domain):
    domain = unicodedata.normalize('NFKC', domain).translate(homoglyph_map)
    tokens = re.split(r'[.\-_/]+', domain)
    return [t for t in tokens if t and not t.isdigit()]

def domain_brand_distance(tokens):
    min_dist = float('inf')
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

def has_ip_address(domain):
    return bool(re.match(r"^\d{1,3}(\.\d{1,3}){3}$", domain))

def count_subdomains(domain):
    return max(len(domain.split(".")) - 2, 0)

def get_domain_age_days(domain):
    try:
        w = whois.whois(domain)
        cd = w.creation_date
        if isinstance(cd, list):
            cd = cd[0]
        if not cd:
            return -1
        return max((datetime.now(timezone.utc) - cd).days, 0)
    except Exception:
        return -1

def extract_url_heuristics(url):
    domain = extract_domain(url)
    age = get_domain_age_days(domain)
    return {
        "url_length": len(url),
        "subdomain_count": count_subdomains(domain),
        "hyphen_count": url.count("-"),
        "has_ip": int(has_ip_address(domain)),
        "has_at": int("@" in url),
        "domain_age_days": age,
        "is_new_domain": int(0 <= age < 30),
        "is_very_new_domain": int(0 <= age < 7)
    }

def path_suspicion_score(url):
    path = extract_path(url)
    tokens = re.split(r"[\/\-_]", path)
    hits = sum(1 for t in tokens if t in suspicious_paths)
    return hits, int(hits > 0)

def shannon_entropy(s):
    if not s:
        return 0.0
    probs = [s.count(c) / len(s) for c in set(s)]
    return -sum(p * math.log2(p) for p in probs)

data["domain"] = data["url"].apply(extract_domain)
data["tokens"] = data["domain"].apply(normalize_domain)
data["entropy"] = data["domain"].apply(shannon_entropy)
data["is_idn"] = data["domain"].apply(is_punycode)

data[["min_lev", "suspicious_count"]] = pd.DataFrame(
    data["tokens"].apply(domain_brand_distance).tolist(),
    index=data.index
)

data["domain_risk"] = (
    (data["suspicious_count"] > 0).astype(int) * 2 +
    (data["min_lev"] <= 1).astype(int)
)

heuristics = data["url"].apply(extract_url_heuristics).apply(pd.Series)
data = pd.concat([data, heuristics], axis=1)

data[["path_hits", "has_suspicious_path"]] = pd.DataFrame(
    data["url"].apply(path_suspicion_score).tolist(),
    index=data.index
)

vectorizer = TfidfVectorizer(
    analyzer="char",
    ngram_range=(3, 5),
    max_features=5000
)

X_tfidf = vectorizer.fit_transform(data["url"]).toarray()

X = np.hstack((
    X_tfidf,
    data.drop(columns=["url", "domain", "tokens", "label"]).values
))

y = data["label"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

model = RandomForestClassifier(
    n_estimators=200,
    random_state=42,
    n_jobs=-1,
    class_weight={0: 1, 1: 2}
)

model.fit(X_train, y_train)

print(classification_report(
    y_test,
    model.predict(X_test),
    zero_division=0
))

joblib.dump(model, "phishing_model.pkl")
joblib.dump(vectorizer, "tfidf_vectorizer.pkl")
