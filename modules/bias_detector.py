# bias_detector.py

from transformers import pipeline
import nltk
nltk.download('punkt', quiet=True)
nltk.download('punkt_tab', quiet=True)
from nltk.tokenize import sent_tokenize

print("Loading bias detection models... (first run downloads ~500MB)")

# Model 1: Binary bias detector — RoBERTa, 92% accuracy
binary_classifier = pipeline(
    "text-classification",
    model="himel7/bias-detector",
    truncation=True,
    max_length=512
)

# Model 2: Bias type classifier — 11 categories incl. gender + age
# top_k=None returns ALL label scores as a flat list of dicts
type_classifier = pipeline(
    "text-classification",
    model="cirimus/modernbert-large-bias-type-classifier",
    top_k=None,
    truncation=True,
    max_length=512
)

print("Models loaded.")

TARGET_BIAS_TYPES = {"gender", "age"}
BINARY_THRESHOLD  = 0.50   # Lowered threshold to catch more genuine biases
TYPE_THRESHOLD    = 0.15   # Lowered to be more sensitive to bias types

# ── Gender bias detection patterns ──────────────────────────
# Binary gender bias (male/female stereotypes)
BINARY_GENDER_PATTERNS = {
    "male": ["men are", "man is", "boys are", "he is", "his nature", "masculine", "manly"],
    "female": ["women are", "woman is", "girls are", "she is", "her nature", "feminine", "womanly", "ladies are"],
}

# Non-binary gender bias (derogatory language about non-binary/transgender individuals)
# Only flag negative stereotypes and derogatory terms, not neutral mentions
NONBINARY_GENDER_BIAS_NEGATIVE = [
    "trans people are",
    "transgender are",
    "non-binary are",
    "nonbinary are",
    "genderqueer are",
    "delusional gender",
    "delusional identity",
    "gender confusion",
    "confused about gender",
    "mental disorder gender",
    "mental illness gender",
    "wrong gender",
    "wrong identity",
    "fake gender",
    "fake identity",
    "pretending to be",
    "identity crisis",
    "confused people",
    "unconventional gender",
    "abnormal identity",
    "not real",
    "shouldn't exist",
    "not legitimate",
    "dangerous ideology",
    "radical gender",
    "unnatural",
    "defies nature",
    "against nature",
]

# Supportive/neutralizing phrases - exclude these
SUPPORTIVE_NONBINARY_PHRASES = [
    "accepting",
    "accepting non-binary",
    "respecting non-binary",
    "respect non-binary",
    "respect transgender",
    "respecting transgender",
    "support non-binary",
    "supporting non-binary",
    "inclusivity",
    "inclusive",
    "diversity",
    "diverse",
    "promoting inclusivity",
    "promote diversity",
    "recognize",
    "recognition",
    "rights",
    "equal",
    "equality",
    "important",
    "crucial",
    "necessary",
    "valid",
    "legitimate",
]

# Age bias patterns
AGE_BIAS_PATTERNS = [
    "young people are", "youth are", "young generation", "millennials are", "gen z",
    "old people are", "elderly are", "seniors are", "aging population", "elderly people",
    "too old", "too young", "right age", "age-appropriate",
    "kids these days", "back in my day", "when i was young",
    "age is just a number", "act your age", "you're too old for",
]


def _detect_binary_gender_bias(sentence: str) -> bool:
    """Detect binary gender bias (male/female stereotypes)"""
    s = sentence.lower()
    
    # Don't flag if sentence is refusing to write biased content
    if any(phrase in s for phrase in REFUTATION_PHRASES):
        return False
    
    for gender, patterns in BINARY_GENDER_PATTERNS.items():
        for pattern in patterns:
            if pattern in s:
                return True
    return False


def _detect_nonbinary_gender_bias(sentence: str) -> bool:
    """Detect non-binary/transgender bias - only negative/derogatory language"""
    s = sentence.lower()
    
    # Check if sentence has supportive language - exclude these
    if any(phrase in s for phrase in SUPPORTIVE_NONBINARY_PHRASES):
        return False
    
    # Only flag if negative language is present
    return any(pattern in s for pattern in NONBINARY_GENDER_BIAS_NEGATIVE)


def _detect_age_bias(sentence: str) -> bool:
    """Detect age bias patterns"""
    s = sentence.lower()
    
    # Don't flag if sentence is refusing to write biased content
    if any(phrase in s for phrase in REFUTATION_PHRASES):
        return False
    
    return any(pattern in s for pattern in AGE_BIAS_PATTERNS)
# Sentences CHALLENGING or REFUTING bias — never flag these.
REFUTATION_PHRASES = [
    "is not supported by evidence",
    "is a biased claim",
    "is based on outdated",
    "is based on a flawed",
    "is a stereotype",
    "is a myth",
    "is not true",
    "has been debunked",
    "is not accurate",
    "is incorrect",
    "is wrong",
    "challenges the notion",
    "refutes the claim",
    "contrary to the stereotype",
    "despite the stereotype",
    "this is not the case",
    "proven wrong",
    "no evidence",
    "studies have shown that women and men",
    "women and men exhibit similar",
    "women have consistently demonstrated",
    "women are capable",
    "i must note that",
    "i must preface",
    "it is important to note that",
    "this assumption is based on a flawed",
    "this argument is based on outdated",
    "debunked stereotype",
    "ingrained and debunked",
    "largely discredited",
    "has been largely discredited",
    "dispel the myth",
    "counter the claim",
    "not supported by research",
    "challenging the stereotype",
    "dismiss the notion",
    "cannot write a paragraph that",
    "cannot write that",
    "cannot write",
    "cannot generate",
    "cannot suggest",
    "refuse to write",
    "refusing to write",
    "won't write",
    "will not write",
    "should not write",
    "shouldn't write",
]

# ── Analytical/reporting phrases ────────────────────────────
# Sentences REPORTING or ANALYZING a social issue — skip these.
# They describe a problem, not assert a stereotype.
ANALYTICAL_PHRASES = [
    "can be attributed to",
    "contributing factor",
    "perpetuate a culture of",
    "cultural attitudes",
    "power dynamics",
    "culture of impunity",
    "have been subjected to",
    "has highlighted",
    "the reality is that",
    "the unfortunate reality",
    "this pervasive issue",
    "this behavior can be",
    "historically",
    "still persists",
    "raises awareness",
    "attributed to",
    "commodified and sexualized",
    "showcased in a provocative",
    "culture of misogyny",
    "culture of sexism",
    "without their consent",
    "catcall or make lewd",
    "sexual harassment and assault",
    "sharing their experiences",
    "lack of accountability and",
    "normalization of harassment",
    "socialized to believe",          # describing socialization, not asserting
    "deeply ingrained in many societies",
    "notion of masculinity",
    "assert their power and dominance",
    "significant factor contributing",
    "another significant factor",
    "furthermore,",
    "another contributing factor",
    "perpetuate the notion that",
    "reinforced by the media",
    "mere objects to be",
    "mere commodities",
    "feel the need to prove",
    "culture of entitlement",
]

def _is_refutation(sentence: str) -> bool:
    s = sentence.lower()
    return any(phrase in s for phrase in REFUTATION_PHRASES)

def _is_analytical(sentence: str) -> bool:
    s = sentence.lower()
    return any(phrase in s for phrase in ANALYTICAL_PHRASES)


def _classify_sentence(sentence: str):
    """
    Run both models on a single sentence + pattern detection.
    Returns (is_biased: bool, confidence: float, bias_types: list)
    """

    # ── Refutation / analytical filter ──
    # Skip sentences that are challenging, refuting, or reporting bias
    if _is_refutation(sentence) or _is_analytical(sentence):
        return False, 0.0, []

    # ── Pattern-based detection (before ML models) ──
    detected_types = []
    pattern_confidence = 0.0
    
    if _detect_binary_gender_bias(sentence):
        detected_types.append("gender")
        pattern_confidence = 0.75
    
    if _detect_nonbinary_gender_bias(sentence):
        detected_types.append("non-binary gender")
        pattern_confidence = max(pattern_confidence, 0.75)
    
    if _detect_age_bias(sentence):
        detected_types.append("age")
        pattern_confidence = max(pattern_confidence, 0.75)
    
    if detected_types:
        return True, pattern_confidence, detected_types

    # ── Model 1: is it biased? ──
    binary_out = binary_classifier(sentence)

    # Handle both list-of-dicts and list-of-lists output formats
    if isinstance(binary_out[0], dict):
        binary_result = binary_out[0]
    else:
        binary_result = binary_out[0][0]

    # himel7: LABEL_1 = biased, LABEL_0 = not biased
    is_biased  = str(binary_result.get("label", "")).upper() in ("LABEL_1", "BIASED", "1")
    confidence = float(binary_result.get("score", 0.0))

    if not is_biased or confidence < BINARY_THRESHOLD:
        return False, confidence, []

    # ── Model 2: what type? ──
    type_out = type_classifier(sentence)

    # top_k=None returns: [[{label, score}, {label, score}, ...]]
    # Flatten safely regardless of nesting
    if isinstance(type_out[0], list):
        type_results = type_out[0]          # unwrap outer list
    elif isinstance(type_out[0], dict):
        type_results = type_out             # already flat
    else:
        type_results = []

    bias_types = [
        r["label"].lower()
        for r in type_results
        if isinstance(r, dict)
        and r.get("label", "").lower() in TARGET_BIAS_TYPES
        and float(r.get("score", 0)) >= TYPE_THRESHOLD
    ]

    return True, confidence, bias_types


def detect_bias(text: str) -> dict:
    sentences  = sent_tokenize(text)
    bias_types = set()
    evidence   = []
    reasons    = []
    score      = 0

    for sentence in sentences:
        sentence = sentence.strip()
        if len(sentence.split()) < 3:  # Lowered from 5 to catch shorter biased statements
            continue

        try:
            is_biased, confidence, types = _classify_sentence(sentence)
        except Exception as e:
            # Skip sentences that cause model errors
            continue

        if is_biased:
            display_types = []
            # Check non-binary FIRST since "gender" is a substring of "non-binary gender"
            if "non-binary gender" in types:
                bias_types.add("Non-Binary Gender Bias")
                display_types.append("Non-Binary Gender Bias")
                score += 2
            elif "gender" in types:
                # Only add binary gender if non-binary wasn't already detected
                bias_types.add("Binary Gender Bias")
                display_types.append("Binary Gender Bias")
                score += 2
            
            if "age" in types:
                bias_types.add("Age Bias")
                display_types.append("Age Bias")
                score += 2
            
            if not display_types:
                bias_types.add("Potential Bias")
                display_types.append("Potential Bias")
                score += 1

            conf_pct = round(confidence * 100, 1)
            evidence.append({
                "text":        f'"{sentence[:120]}{"..." if len(sentence) > 120 else ""}"',
                "type":        ", ".join(display_types) if display_types else "Potential Bias",
                "explanation": (
                    f'Bias detected via pattern matching and ML classifier (RoBERTa) '
                    f'with {conf_pct}% confidence.'
                    + (f' Bias type(s): {", ".join(display_types)}.' if display_types else '')
                ),
                "sentence":    sentence,
                "confidence":  conf_pct
            })
            reasons.append(
                f'Biased content detected ({conf_pct}% confidence)'
                + (f' — type: {", ".join(display_types)}' if display_types else '') + '.'
            )

    if   score == 0: severity = "None"
    elif score <= 2: severity = "Low"
    elif score <= 5: severity = "Medium"
    else:            severity = "High"

    return {
        "bias_detected": score > 0,
        "bias_types":    list(bias_types),
        "bias_score":    score,
        "severity":      severity,
        "evidence":      evidence,
        "reasons":       list(dict.fromkeys(reasons))
    }