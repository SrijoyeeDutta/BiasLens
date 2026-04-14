# test_logger.py
# ─────────────────────────────────────────────────────────────
# Saves every test case (prompt + generated text + bias results)
# to a local JSON file for academic reporting.
# ─────────────────────────────────────────────────────────────

import json
import os
from datetime import datetime

LOG_FILE = "test_cases.json"


def _load_log() -> list:
    if os.path.exists(LOG_FILE):
        try:
            with open(LOG_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return []
    return []


def _save_log(data: list):
    with open(LOG_FILE, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def save_test_case(
    prompt: str,
    generated_text: str,
    layer1_result: dict,
    layer2_result: dict,
    mitigation_result: dict = None
) -> dict:
    """
    Save a complete test case to test_cases.json.
    Returns the saved test case dict.
    """

    # ── Compute combined severity ──────────────────────────
    sev_order = {"None": 0, "Low": 1, "Medium": 2, "High": 3}
    l1_sev = layer1_result.get("severity", "None")
    l2_sev = layer2_result.get("overall_severity", "None")
    combined_sev = max(l1_sev, l2_sev, key=lambda s: sev_order.get(s, 0))

    # ── All bias types found ────────────────────────────────
    all_types = list(set(
        layer1_result.get("bias_types", []) +
        [b.get("bias_type", "") for b in layer2_result.get("biases_found", [])]
    ))

    # ── Build test case record ──────────────────────────────
    test_case = {
        "id":               None,           # filled below
        "timestamp":        datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "prompt":           prompt,
        "generated_text":   generated_text,

        "layer1": {
            "bias_detected":  layer1_result.get("bias_detected", False),
            "bias_score":     layer1_result.get("bias_score", 0),
            "severity":       l1_sev,
            "bias_types":     layer1_result.get("bias_types", []),
            "evidence":       [
                {
                    "type":        e.get("type", ""),
                    "text":        e.get("text", ""),
                    "explanation": e.get("explanation", ""),
                    "confidence":  e.get("confidence", 0)
                }
                for e in layer1_result.get("evidence", [])
            ],
            "reasons": layer1_result.get("reasons", [])
        },

        "layer2": {
            "overall_severity":   l2_sev,
            "overall_assessment": layer2_result.get("overall_assessment", ""),
            "biases_found": [
                {
                    "bias_type":   b.get("bias_type", ""),
                    "title":       b.get("title", ""),
                    "evidence":    b.get("evidence", ""),
                    "explanation": b.get("explanation", ""),
                    "severity":    b.get("severity", "")
                }
                for b in layer2_result.get("biases_found", [])
            ]
        },

        "summary": {
            "combined_severity":    combined_sev,
            "all_bias_types":       all_types,
            "total_findings":       len(layer1_result.get("evidence", [])) + len(layer2_result.get("biases_found", [])),
            "layer1_findings":      len(layer1_result.get("evidence", [])),
            "layer2_findings":      len(layer2_result.get("biases_found", [])),
            "bias_detected":        combined_sev != "None",
            "bias_reduced":         mitigation_result.get("bias_reduced", False) if mitigation_result else False
        }
    }

    # ── Store full mitigation output as layer3 ──────────────
    if mitigation_result:
        val = mitigation_result.get("validation") or {}
        test_case["layer3"] = {
            "bias_reduced":   mitigation_result.get("bias_reduced", False),
            "summary":        mitigation_result.get("summary", ""),
            "original_text":  mitigation_result.get("original_text", ""),
            "final_text":     mitigation_result.get("final_text", ""),
            "rule_changes":   mitigation_result.get("rule_changes", []),
            "spacy_changes":  mitigation_result.get("spacy_changes", []),
            "bert_changes":   mitigation_result.get("bert_changes", []),
            "cda_changes":    mitigation_result.get("cda_changes", []),
            "wordnet_changes":mitigation_result.get("wordnet_changes", []),
            "svo_changes":    mitigation_result.get("svo_changes", []),
            "passive_flags":  mitigation_result.get("passive_flags", []),
            "double_standards":mitigation_result.get("double_standards", []),
            "stage1_text":    mitigation_result.get("stage1_text", ""),
            "stage2_text":    mitigation_result.get("stage2_text", ""),
            "stage3_text":    mitigation_result.get("stage3_text", ""),
            "stage4_text":    mitigation_result.get("stage4_text", ""),
            "stage5_text":    mitigation_result.get("stage5_text", ""),
            "stage6_text":    mitigation_result.get("stage6_text", ""),
            "validation": {
                "meaning_similarity":       val.get("meaning_similarity"),
                "meaning_preserved":        val.get("meaning_preserved"),
                "overall_bias_reduction":   val.get("overall_bias_reduction"),
                "neutrality_score":         val.get("neutrality_score"),
                "orig_gender_sensitivity":  val.get("orig_gender_sensitivity"),
                "orig_age_sensitivity":     val.get("orig_age_sensitivity"),
                "mitig_gender_sensitivity": val.get("mitig_gender_sensitivity"),
                "mitig_age_sensitivity":    val.get("mitig_age_sensitivity"),
                "warning":                  val.get("warning", "")
            }
        }

    # ── Append to log ───────────────────────────────────────
    log = _load_log()
    test_case["id"] = len(log) + 1
    log.append(test_case)
    _save_log(log)

    return test_case


def load_all_test_cases() -> list:
    return _load_log()


def delete_test_case(case_id: int):
    log = _load_log()
    log = [c for c in log if c.get("id") != case_id]
    _save_log(log)


def clear_all_test_cases():
    _save_log([])


def get_accuracy_stats() -> dict:
    """
    Compute basic accuracy statistics across all saved test cases.
    """
    log = _load_log()
    if not log:
        return {}

    total       = len(log)
    biased      = sum(1 for c in log if c["summary"]["bias_detected"])
    not_biased  = total - biased

    sev_counts  = {"None": 0, "Low": 0, "Medium": 0, "High": 0}
    type_counts = {}
    l1_hits     = 0
    l2_hits     = 0
    both_hits   = 0

    for c in log:
        sev = c["summary"]["combined_severity"]
        sev_counts[sev] = sev_counts.get(sev, 0) + 1

        for t in c["summary"]["all_bias_types"]:
            type_counts[t] = type_counts.get(t, 0) + 1

        l1 = c["layer1"]["bias_detected"]
        l2 = len(c["layer2"]["biases_found"]) > 0

        if l1: l1_hits += 1
        if l2: l2_hits += 1
        if l1 and l2: both_hits += 1

    return {
        "total_test_cases":         total,
        "biased_cases":             biased,
        "clean_cases":              not_biased,
        "bias_detection_rate":      round(biased / total * 100, 1) if total else 0,
        "severity_distribution":    sev_counts,
        "bias_type_frequency":      dict(sorted(type_counts.items(), key=lambda x: x[1], reverse=True)),
        "layer1_detection_rate":    round(l1_hits / total * 100, 1) if total else 0,
        "layer2_detection_rate":    round(l2_hits / total * 100, 1) if total else 0,
        "both_layers_agreement":    round(both_hits / total * 100, 1) if total else 0,
    }
