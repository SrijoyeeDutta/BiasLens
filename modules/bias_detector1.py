# bias_detector_roberta_only.py
# ─────────────────────────────────────────────────────────────
# ABLATION STUDY VERSION — RoBERTa ONLY (no ModernBERT)
#
# Purpose: Compare accuracy of RoBERTa alone vs RoBERTa+ModernBERT
# Use this file by replacing bias_detector.py temporarily,
# run your test cases, export results, then switch back.
#
# What is REMOVED vs full version:
#   ✗ cirimus/modernbert-large-bias-type-classifier (type classifier)
#
# What is KEPT:
#   ✓ himel7/bias-detector (RoBERTa binary classifier)
#   ✓ Pattern matching (binary gender, non-binary gender, age)
#   ✓ Refutation/analytical phrase filter
#
# Since ModernBERT told us the TYPE (gender/age), without it
# we fall back to "Potential Bias" for ML-detected sentences
# unless pattern matching already identified the type.
# ─────────────────────────────────────────────────────────────

from transformers import pipeline
import nltk
nltk.download('punkt', quiet=True)
nltk.download('punkt_tab', quiet=True)
from nltk.tokenize import sent_tokenize

print("Loading RoBERTa-only bias detection model...")

# ── ONLY RoBERTa — ModernBERT is NOT loaded ──────────────────
binary_classifier = pipeline(
    "text-classification",
    model="himel7/bias-detector",
    truncation=True,
    max_length=512
)

print("RoBERTa model loaded. (ModernBERT disabled for ablation study)")

BINARY_THRESHOLD = 0.50

# ── Pattern lists (same as full version) ─────────────────────

BINARY_GENDER_PATTERNS = {
    "male":   ["men are", "man is", "boys are", "he is", "his nature", "masculine", "manly"],
    "female": ["women are", "woman is", "girls are", "she is", "her nature", "feminine", "womanly", "ladies are"],
}

NONBINARY_GENDER_BIAS_NEGATIVE = [
    "trans people are", "transgender are", "non-binary are", "nonbinary are",
    "genderqueer are", "delusional gender", "delusional identity", "gender confusion",
    "confused about gender", "mental disorder gender", "mental illness gender",
    "wrong gender", "wrong identity", "fake gender", "fake identity",
    "pretending to be", "identity crisis", "confused people", "unconventional gender",
    "abnormal identity", "not real", "shouldn't exist", "not legitimate",
    "dangerous ideology", "radical gender", "unnatural", "defies nature", "against nature",
]

SUPPORTIVE_NONBINARY_PHRASES = [
    "accepting", "accepting non-binary", "respecting non-binary", "respect non-binary",
    "respect transgender", "respecting transgender", "support non-binary",
    "supporting non-binary", "inclusivity", "inclusive", "diversity", "diverse",
    "promoting inclusivity", "promote diversity", "recognize", "recognition",
    "rights", "equal", "equality", "important", "crucial", "necessary", "valid", "legitimate",
]

AGE_BIAS_PATTERNS = [
    "young people are", "youth are", "young generation", "millennials are", "gen z",
    "old people are", "elderly are", "seniors are", "aging population", "elderly people",
    "too old", "too young", "right age", "age-appropriate",
    "kids these days", "back in my day", "when i was young",
    "age is just a number", "act your age", "you're too old for",
]

REFUTATION_PHRASES = [
    "is not supported by evidence", "is a biased claim", "is based on outdated",
    "is based on a flawed", "is a stereotype", "is a myth", "is not true",
    "has been debunked", "is not accurate", "is incorrect", "is wrong",
    "challenges the notion", "refutes the claim", "contrary to the stereotype",
    "despite the stereotype", "this is not the case", "proven wrong", "no evidence",
    "studies have shown that women and men", "women and men exhibit similar",
    "women have consistently demonstrated", "women are capable",
    "i must note that", "i must preface", "it is important to note that",
    "this assumption is based on a flawed", "this argument is based on outdated",
    "debunked stereotype", "ingrained and debunked", "largely discredited",
    "has been largely discredited", "dispel the myth", "counter the claim",
    "not supported by research", "challenging the stereotype", "dismiss the notion",
    "cannot write a paragraph that", "cannot write that", "cannot write",
    "cannot generate", "cannot suggest", "refuse to write", "refusing to write",
    "won't write", "will not write", "should not write", "shouldn't write",
]

ANALYTICAL_PHRASES = [
    "can be attributed to", "contributing factor", "perpetuate a culture of",
    "cultural attitudes", "power dynamics", "culture of impunity",
    "have been subjected to", "has highlighted", "the reality is that",
    "the unfortunate reality", "this pervasive issue", "this behavior can be",
    "historically", "still persists", "raises awareness", "attributed to",
    "commodified and sexualized", "showcased in a provocative", "culture of misogyny",
    "culture of sexism", "without their consent", "catcall or make lewd",
    "sexual harassment and assault", "sharing their experiences",
    "lack of accountability and", "normalization of harassment",
    "socialized to believe", "deeply ingrained in many societies",
    "notion of masculinity", "assert their power and dominance",
    "significant factor contributing", "another significant factor",
    "furthermore,", "another contributing factor", "perpetuate the notion that",
    "reinforced by the media", "mere objects to be", "mere commodities",
    "feel the need to prove", "culture of entitlement",
]


def _is_refutation(sentence: str) -> bool:
    s = sentence.lower()
    return any(phrase in s for phrase in REFUTATION_PHRASES)

def _is_analytical(sentence: str) -> bool:
    s = sentence.lower()
    return any(phrase in s for phrase in ANALYTICAL_PHRASES)

def _detect_binary_gender_bias(sentence: str) -> bool:
    s = sentence.lower()
    if _is_refutation(s): return False
    for patterns in BINARY_GENDER_PATTERNS.values():
        if any(p in s for p in patterns):
            return True
    return False

def _detect_nonbinary_gender_bias(sentence: str) -> bool:
    s = sentence.lower()
    if any(p in s for p in SUPPORTIVE_NONBINARY_PHRASES): return False
    return any(p in s for p in NONBINARY_GENDER_BIAS_NEGATIVE)

def _detect_age_bias(sentence: str) -> bool:
    s = sentence.lower()
    if _is_refutation(s): return False
    return any(p in s for p in AGE_BIAS_PATTERNS)


def _classify_sentence(sentence: str):
    """
    RoBERTa-only classification.
    Type is determined by pattern matching only — ModernBERT not used.
    """

    # ── Filter ──────────────────────────────────────────────
    if _is_refutation(sentence) or _is_analytical(sentence):
        return False, 0.0, []

    # ── Pattern matching for type ───────────────────────────
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

    # ── RoBERTa: is it biased? ──────────────────────────────
    binary_out = binary_classifier(sentence)

    if isinstance(binary_out[0], dict):
        binary_result = binary_out[0]
    else:
        binary_result = binary_out[0][0]

    is_biased  = str(binary_result.get("label", "")).upper() in ("LABEL_1", "BIASED", "1")
    confidence = float(binary_result.get("score", 0.0))

    if not is_biased or confidence < BINARY_THRESHOLD:
        return False, confidence, []

    # ── NO ModernBERT — type is unknown, return empty list ──
    # The detect_bias() function will mark this as "Potential Bias"
    return True, confidence, []


def detect_bias(text: str) -> dict:
    sentences  = sent_tokenize(text)
    bias_types = set()
    evidence   = []
    reasons    = []
    score      = 0

    for sentence in sentences:
        sentence = sentence.strip()
        if len(sentence.split()) < 3:
            continue

        try:
            is_biased, confidence, types = _classify_sentence(sentence)
        except Exception:
            continue

        if is_biased:
            display_types = []

            if "non-binary gender" in types:
                bias_types.add("Non-Binary Gender Bias")
                display_types.append("Non-Binary Gender Bias")
                score += 2
            elif "gender" in types:
                bias_types.add("Binary Gender Bias")
                display_types.append("Binary Gender Bias")
                score += 2

            if "age" in types:
                bias_types.add("Age Bias")
                display_types.append("Age Bias")
                score += 2

            if not display_types:
                # RoBERTa says biased but no pattern found type
                # Without ModernBERT we can't classify — mark as Potential
                bias_types.add("Potential Bias")
                display_types.append("Potential Bias")
                score += 1

            conf_pct = round(confidence * 100, 1)
            evidence.append({
                "text":        f'"{sentence[:120]}{"..." if len(sentence) > 120 else ""}"',
                "type":        ", ".join(display_types) if display_types else "Potential Bias",
                "explanation": (
                    f'RoBERTa classifier flagged this sentence as biased '
                    f'with {conf_pct}% confidence. '
                    f'[ABLATION MODE: ModernBERT type classifier disabled]'
                    + (f' Detected type(s) via pattern matching: {", ".join(display_types)}.' if display_types else ' Type unknown — ModernBERT not used.')
                ),
                "sentence":    sentence,
                "confidence":  conf_pct
            })
            reasons.append(
                f'RoBERTa flagged ({conf_pct}% confidence)'
                + (f' — type via pattern: {", ".join(display_types)}' if display_types else ' — type unknown (ModernBERT disabled)') + '.'
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