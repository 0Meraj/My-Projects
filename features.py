import re
import math
import unicodedata
import numpy as np
import whois
import ipaddress
import tldextract
from datetime import datetime, timezone
from urllib.parse import urlparse
from rapidfuzz.distance import Levenshtein

# ---------------------------
# CONSTANTS
# ---------------------------

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

# Manual feature order (CRITICAL — must never change)
MANUAL_FEATURE_ORDER = [
    "entropy",
    "is_idn",
    "min_lev",
    "suspicious_count",
    "path_hits",
    "has_suspicious_path",
    "domain_risk",
    "url_length",
    "subdomain_count",
    "hyphen_count",
    "has_ip",
    "has_at",
    "domain_age_days",
    "is_new_domain",
    "is_very_new_domain"
]

# ---------------------------
# CORE HELPERS
# ---------------------------

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

def shannon_entropy(s):
    if not s:
        return 0.0
    probs = [s.count(c) / len(s) for c in set(s)]
    return -sum(p * math.log2(p) for p in probs)

def domain_brand_distance(tokens):
    min_dist = float("inf")
    suspicious = 0
    closest_brand = None

    for token in tokens:
        for brand in protected_brands:
            d = Levenshtein.distance(token, brand)
            if d < min_dist:
                min_dist = d
                closest_brand = brand
            if d <= 1 and token != brand:
                suspicious += 1

        if token in suspicious_keywords:
            suspicious += 1

    return min_dist, suspicious, closest_brand

def path_suspicion_score(url):
    tokens = re.split(r"[\/\-_]", extract_path(url))
    hits = sum(1 for t in tokens if t in suspicious_paths)
    matched = [t for t in tokens if t in suspicious_paths]
    return hits, int(hits > 0), matched

def has_ip_address(domain):
    try:
        ipaddress.ip_address(domain)
        return True
    except ValueError:
        return False

def count_subdomains(domain):
    ext = tldextract.extract(domain)
    if not ext.subdomain:
        return 0
    return len(ext.subdomain.split("."))

def get_domain_age_days(domain):
    try:
        w = whois.whois(domain)
        cd = w.creation_date
        if isinstance(cd, list):
            cd = cd[0]
        if not cd:
            return -1
        if isinstance(cd, str):
            try:
                cd = datetime.fromisoformat(cd)
            except ValueError:
                return -1
        if cd.tzinfo is None:
            cd = cd.replace(tzinfo=timezone.utc)
        return max((datetime.now(timezone.utc) - cd).days, 0)
    except Exception:
        return -1

# ---------------------------
# MAIN FEATURE BUILDER
# ---------------------------

def build_manual_features(url):
    """
    Returns:
        features_array  — np.array of shape (1, len(MANUAL_FEATURE_ORDER))
        flagged         — list of (key, human_label, severity) for every triggered heuristic,
                          sorted by severity descending. Empty list for clean URLs.
    """
    domain = extract_domain(url)
    tokens = normalize_domain(domain)

    min_lev, suspicious_count, closest_brand = domain_brand_distance(tokens)
    entropy = shannon_entropy(domain)
    path_hits, has_suspicious_path, matched_paths = path_suspicion_score(url)
    age = -1

    feature_dict = {
        "entropy":            entropy,
        "is_idn":             int(is_punycode(domain)),
        "min_lev":            min_lev,
        "suspicious_count":   suspicious_count,
        "path_hits":          path_hits,
        "has_suspicious_path":has_suspicious_path,
        "domain_risk":        (suspicious_count > 0) * 2 + (min_lev <= 1),
        "url_length":         len(url),
        "subdomain_count":    count_subdomains(domain),
        "hyphen_count":       url.count("-"),
        "has_ip":             int(has_ip_address(domain)),
        "has_at":             int("@" in url),
        "domain_age_days":    age,
        "is_new_domain":      int(0 <= age < 30),
        "is_very_new_domain": int(0 <= age < 7),
    }

    # --- Build flagged heuristics list ---
    # Each entry: (key, human-readable label, severity weight)
    flagged = []

    if feature_dict["has_ip"]:
        flagged.append(("has_ip", "Domain is a raw IP address", 10))

    if feature_dict["has_at"]:
        flagged.append(("has_at", "URL contains an @ symbol", 9))

    if min_lev <= 1 and closest_brand:
        flagged.append(("min_lev", f"Domain closely mimics '{closest_brand}'", 9))
    elif min_lev <= 2 and closest_brand:
        flagged.append(("min_lev", f"Domain resembles '{closest_brand}'", 6))

    if feature_dict["is_idn"]:
        flagged.append(("is_idn", "Uses internationalised (punycode) domain — possible homoglyph attack", 8))

    if suspicious_count >= 2:
        flagged.append(("suspicious_count", f"Multiple suspicious keywords detected ({suspicious_count})", 7))
    elif suspicious_count == 1:
        flagged.append(("suspicious_count", "Suspicious keyword found in domain", 5))

    if feature_dict["is_very_new_domain"]:
        flagged.append(("is_very_new_domain", "Domain registered within the last 7 days", 8))
    elif feature_dict["is_new_domain"]:
        flagged.append(("is_new_domain", "Domain registered within the last 30 days", 6))

    if has_suspicious_path:
        path_str = ", ".join(matched_paths[:2])
        flagged.append(("has_suspicious_path", f"Suspicious path segment detected: '{path_str}'", 5))

    if feature_dict["hyphen_count"] >= 3:
        flagged.append(("hyphen_count", f"Unusually high hyphen count ({feature_dict['hyphen_count']})", 4))

    if feature_dict["subdomain_count"] >= 3:
        flagged.append(("subdomain_count", f"Excessive subdomain depth ({feature_dict['subdomain_count']} levels)", 4))

    if entropy > 4.0:
        flagged.append(("entropy", f"High domain entropy ({entropy:.2f}) — randomised or obfuscated string", 3))

    if feature_dict["url_length"] > 100:
        flagged.append(("url_length", f"Unusually long URL ({feature_dict['url_length']} characters)", 3))

    # Sort by severity descending
    flagged.sort(key=lambda x: x[2], reverse=True)

    features_array = np.array([[feature_dict[col] for col in MANUAL_FEATURE_ORDER]])
    return features_array, flagged