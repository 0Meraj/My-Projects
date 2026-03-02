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

    for token in tokens:
        for brand in protected_brands:
            d = Levenshtein.distance(token, brand)
            min_dist = min(min_dist, d)
            if d <= 1 and token != brand:
                suspicious += 1

        if token in suspicious_keywords:
            suspicious += 1

    return min_dist, suspicious

def path_suspicion_score(url):
    tokens = re.split(r"[\/\-_]", extract_path(url))
    hits = sum(1 for t in tokens if t in suspicious_paths)
    return hits, int(hits > 0)

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

        # Handle string dates (some WHOIS providers return strings)
        if isinstance(cd, str):
            try:
                cd = datetime.fromisoformat(cd)
            except ValueError:
                return -1

        # Make timezone-aware if naive
        if cd.tzinfo is None:
            cd = cd.replace(tzinfo=timezone.utc)

        return max((datetime.now(timezone.utc) - cd).days, 0)

    except Exception:
        return -1

# ---------------------------
# MAIN FEATURE BUILDER
# ---------------------------

def build_manual_features(url):
    domain = extract_domain(url)
    tokens = normalize_domain(domain)

    min_lev, suspicious_count = domain_brand_distance(tokens)
    entropy = shannon_entropy(domain)
    path_hits, has_suspicious_path = path_suspicion_score(url)
    age = -1

    feature_dict = {
        "entropy": entropy,
        "is_idn": int(is_punycode(domain)),
        "min_lev": min_lev,
        "suspicious_count": suspicious_count,
        "path_hits": path_hits,
        "has_suspicious_path": has_suspicious_path,
        "domain_risk": (suspicious_count > 0) * 2 + (min_lev <= 1),
        "url_length": len(url),
        "subdomain_count": count_subdomains(domain),
        "hyphen_count": url.count("-"),
        "has_ip": int(has_ip_address(domain)),
        "has_at": int("@" in url),
        "domain_age_days": age,
        "is_new_domain": int(0 <= age < 30),
        "is_very_new_domain": int(0 <= age < 7)
    }

    return np.array([[feature_dict[col] for col in MANUAL_FEATURE_ORDER]])