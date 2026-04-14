# bias_mitigator.py
# ─────────────────────────────────────────────────────────────
# Bias Mitigation — LLM-Driven Targeted Rewriting
#
# Uses the actual detected bias findings from Layer 1 (ML) and
# Layer 2 (LLM contextual analysis) to drive precise mitigation.
#
# Both EXPLICIT and IMPLICIT/CONTEXTUAL biases are mitigated:
#   - Explicit: direct stereotype statements ("men are naturally...")
#   - Implicit: occupational coding, trait essentialism,
#               invisibility bias, double standards, name-role
#               stereotyping, implicit age framing
#
# The mitigator sends the original text + ALL detected findings
# to Llama 3.3 70B with a detailed prompt that instructs it to:
#   1. Fix each detected finding specifically
#   2. Preserve the original meaning and facts
#   3. Return a changelog of every change made
# ─────────────────────────────────────────────────────────────

import os
import json
import re
from dotenv import load_dotenv
from groq import Groq

# ─────────────────────────────────────────
# Groq API — Free tier, no credit card needed
# 1. Sign up free at https://console.groq.com
# 2. Go to API Keys → Create API Key → copy it
# 3. Paste your key in .env: GROQ_API_KEY=your_key_here
# ─────────────────────────────────────────

load_dotenv()
api_key = os.getenv("GROQ_API_KEY")
client = Groq(api_key=api_key)



# ── Build mitigation prompt from detected findings ────────────

def _build_findings_summary(rule_result: dict, llm_result: dict) -> str:
    """
    Converts Layer 1 + Layer 2 findings into a clear instruction
    list for the LLM mitigator to act on.
    """
    lines = []

    # Layer 1 findings
    l1_evidence = rule_result.get("evidence", [])
    if l1_evidence:
        lines.append("LAYER 1 — ML CLASSIFIER FINDINGS (explicit pattern-based bias):")
        for i, e in enumerate(l1_evidence, 1):
            lines.append(
                f"  {i}. Type: {e.get('type', 'Bias')}\n"
                f"     Sentence: {e.get('sentence', e.get('text', ''))}\n"
                f"     Confidence: {e.get('confidence', '')}%"
            )

    # Layer 2 findings
    l2_biases = llm_result.get("biases_found", [])
    if l2_biases:
        lines.append("\nLAYER 2 — AI CONTEXTUAL ANALYSIS FINDINGS (implicit/contextual bias):")
        for i, b in enumerate(l2_biases, 1):
            lines.append(
                f"  {i}. Type: {b.get('bias_type', 'Bias')} | Severity: {b.get('severity', '')}\n"
                f"     Title: {b.get('title', '')}\n"
                f"     Evidence: \"{b.get('evidence', '')}\"\n"
                f"     Why biased: {b.get('explanation', '')}"
            )

    if not lines:
        return "No specific bias findings provided."

    return "\n".join(lines)


MITIGATION_SYSTEM_PROMPT = """You are an expert bias editor. Your task is to rewrite a given text to remove all gender and age bias while preserving the original meaning, facts, and professional tone.

You will be given:
1. The original biased text
2. A detailed list of ALL detected biases — both explicit (direct stereotypes) and implicit/contextual (subtle patterns like occupational gender coding, trait essentialism, invisibility bias, double standards, name-role stereotyping, implicit age framing)

YOUR JOB:
- Fix EVERY detected bias listed, addressing both explicit AND implicit findings
- Do not just fix the obvious words — fix the underlying assumptions and framing
- Preserve facts, statistics, and the overall message of the text
- Keep the same length and professional tone
- Use gender-neutral language for roles and pronouns, but DO NOT remove gender when it is the subject of comparison.
- Remove unsupported or exaggerated essentialist claims (e.g., "naturally superior")
- Scientifically valid biological differences may be included if presented with qualifiers such as "on average", "tend to", or "may"
- Reframe occupational gender coding — describe professions and skills without assigning them to specific genders
- Fix invisibility bias by acknowledging that all genders can hold any role
- Fix double standards by applying the same criteria to all groups
- Fix trait essentialism by attributing traits to individuals, not to gender/age groups
- Fix implicit age framing by not implying any age group is exclusively better at something
- Prefer neutral, academic openings.
- Avoid vague comparisons like "some individuals are better than others".
- Use phrasing such as: "While leadership effectiveness varies across individuals..." instead of comparative or group-based statements.

PRIORITY RULE (CRITICAL):
When rules conflict, follow this priority order:
1. Preserve the original question and intent
2. Preserve factual accuracy and valid comparisons
3. Remove bias and harmful generalizations
4. Improve tone and neutrality

Never sacrifice factual correctness or question intent in order to remove bias.

OPENING FRAME RULE:
The rewritten text must begin with a neutral, non-biased framing.
It should:
- Avoid biased or absolute comparisons at the start
- Neutral, qualified comparisons are allowed if required by the original question
- Avoid claims of inherent or natural superiority of any group
- Focus on individuals, skills, context, or variability
The opening should be context-aware and natural, NOT fixed or templated.

MEANING PRESERVATION RULE (STRICT):
- Do NOT introduce new topics that are not present in the original text
- Do NOT shift domains (e.g., physical strength → leadership)
- Keep the same subject matter and context
- Only modify biased phrasing, not the core topic
If a sentence is about physical ability, it must remain about physical ability.
If a sentence is about careers, it must remain about careers.

PRECISION RULE:
- Replace biased statements with neutral equivalents WITHOUT making them overly vague
- Avoid generic phrases like "some individuals" unless necessary
- Preserve specificity when it does not introduce bias

QUESTION PRESERVATION RULE:
- If the original text answers a factual or comparative question, the rewritten version must still answer that question.
- Do NOT remove comparisons if they are factually valid.
- Instead, present them with proper context, nuance, and without implying superiority.
For example:
- Allowed: "On average, group A may show X due to biological factors"
- Not allowed: "Group A is naturally better than group B"

ATTRIBUTE RETENTION RULE:
- Do NOT remove key attributes (e.g., gender, age) if they are central to the topic
- If the original text discusses differences between groups, those groups must remain in the rewritten version
- Replace biased framing, NOT the subject itself

FACTUAL BALANCE RULE:
- Preserve scientifically supported information
- Add qualifiers like "on average", "tend to", "may", "varies"
- Emphasize overlap and individual variation
- Avoid absolute or deterministic claims

COMPARISON HANDLING RULE:
When the original text involves comparison between groups:
- Retain the comparison if it is factually grounded
- Add qualifiers (e.g., "on average", "tend to")
- Emphasize overlap and individual variation
- Avoid framing one group as superior

ATTRIBUTE PRESERVATION ENFORCEMENT (CRITICAL):
- If the original text explicitly compares groups (e.g., men vs women), the rewritten text MUST explicitly retain those groups.
- Do NOT replace group references with generic terms like "individuals" when the comparison is central to the topic.
- The subject of comparison must remain visible in the final text.

NORMATIVE QUESTION RULE:
- If the input involves a "should", "is it right", or ethical question,
  the rewritten text must provide a clear, neutral, and reasoned answer.
- The response should affirm fairness, inclusion, and equal opportunity
  where applicable, without using biased assumptions.
- Avoid avoiding the question — provide a direct but professionally framed answer.

HARMFUL CLAIM HANDLING RULE:
- If the original text contains statements that invalidate or question the legitimacy of identities (e.g., "X identities are confusing or unclear"),
  the rewritten version must NOT repeat them as-is.
- Such statements should be reframed in a neutral, respectful way,
  for example by acknowledging differences in awareness or perspectives,
  without endorsing or amplifying the harmful claim.

DIRECT ANSWER COMPLETION RULE:
- If the original input is a "should" or suitability question,
  the rewritten text must include a clear concluding statement
  that directly addresses the question.
- The answer should be framed in terms of fairness, equality,
  and professional standards where appropriate.

CONCLUSION ENFORCEMENT RULE:
- The final paragraph must clearly and explicitly answer the core question.
- The answer should be concise, neutral, and framed in terms of fairness,
  equal opportunity, or evidence-based reasoning.
  
MITIGATION RULES:
1. "Men are naturally more X" → "Individuals can develop X regardless of gender"
2. "Women are more nurturing/emotional" → Remove or rephrase as individual variation
3. "Old workers resist change" → "Some workers prefer established methods"
4. "Young people are reckless/inexperienced" → "Those new to a field are still developing"
5. Occupational gender coding → Describe the role's skills without gendering them
6. Name-role stereotyping → Keep names but remove role-gender assumptions
7. Invisibility bias → Add inclusive language acknowledging diverse participants
8. Double standards → Apply the same framing to all groups equally

OUTPUT FORMAT — respond ONLY with valid JSON, no markdown, no preamble:
{
  "mitigated_text": "the fully rewritten text with ALL biases addressed",
  "changes": [
    {
      "original": "exact phrase from original text",
      "replacement": "what it was changed to",
      "bias_type": "which type of bias was fixed",
      "reason": "brief explanation of why this change was made"
    }
  ],
  "summary": "2-3 sentence summary of what was changed and why"
}"""


def mitigate_bias(text: str, rule_result: dict, llm_result: dict = None) -> dict:
    """
    LLM-driven bias mitigation using all detected findings.

    Args:
        text:        Original generated text
        rule_result: Layer 1 ML detection results
        llm_result:  Layer 2 LLM detection results

    Returns dict with:
        mitigated_text — fully rewritten text
        changes        — list of specific changes made
        summary        — plain English summary
        original_text  — for comparison
        bias_reduced   — bool
    """

    if llm_result is None:
        llm_result = {}

    # ── Check if there's anything to mitigate ────────────────
    has_l1 = bool(rule_result.get("evidence"))
    has_l2 = bool(llm_result.get("biases_found"))

    if not has_l1 and not has_l2:
        return {
            "original_text":  text,
            "mitigated_text": text,
            "final_text":     text,

            "changes":        [],

            # ✅ ADD THESE
            "rule_changes":   [],
            "spacy_changes":  [],
            "bert_changes":   [],
            "cda_changes":    [],
            "wordnet_changes":[],
            "svo_changes":    [],

            "stage1_text": text,
            "stage2_text": text,
            "stage3_text": text,
            "stage4_text": text,
            "stage5_text": text,
            "stage6_text": text,

            "summary":        "No bias detected — no mitigation needed.",
            "bias_reduced":   False,
            "strategy":       "None — text was already neutral"
        }

    # ── Build findings summary for the prompt ────────────────
    findings = _build_findings_summary(rule_result, llm_result)

    # ── Build user message ────────────────────────────────────
    user_message = f"""Please rewrite the following text to remove ALL detected biases.

ORIGINAL TEXT:
{text}

DETECTED BIASES TO FIX:
{findings}

Remember: Fix EVERY finding listed above — both explicit stereotypes AND subtle contextual biases.
Respond ONLY with valid JSON."""

    # ── Call Groq LLM ─────────────────────────────────────────
    try:
        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[
                {"role": "system", "content": MITIGATION_SYSTEM_PROMPT},
                {"role": "user",   "content": user_message}
            ],
            max_tokens=2000,
            temperature=0.1,   # Low temp for consistent, precise rewrites
        )

        raw = response.choices[0].message.content.strip()
        raw = re.sub(r"```json|```", "", raw).strip()

        # Extract JSON safely
        json_start = raw.find("{")
        if json_start > 0:
            raw = raw[json_start:]

        result = json.loads(raw)

        mitigated_text = result.get("mitigated_text", text)
        changes        = result.get("changes", [])
        summary        = result.get("summary", "Mitigation complete.")

        return {
            "original_text":  text,
            "mitigated_text": mitigated_text,
            "final_text":     mitigated_text,

            # ✅ NEW (LLM output)
            "changes": changes,

            # ✅ BACKWARD COMPATIBILITY (for your UI)
            "rule_changes":   [c["original"] + " → " + c["replacement"] for c in changes],
            "spacy_changes":  [],
            "bert_changes":   [],
            "cda_changes":    [],
            "wordnet_changes":[],
            "svo_changes":    [],

            # stage texts (just reuse final text)
            "stage1_text": mitigated_text,
            "stage2_text": mitigated_text,
            "stage3_text": mitigated_text,
            "stage4_text": mitigated_text,
            "stage5_text": mitigated_text,
            "stage6_text": mitigated_text,

            # other fields
            "summary":        summary,
            "bias_reduced":   mitigated_text != text,
            "strategy":       "LLM-targeted rewriting (Llama 3.3 70B)",
            "l1_findings_used": len(rule_result.get("evidence", [])),
            "l2_findings_used": len(llm_result.get("biases_found", [])),
            "total_changes":  len(changes)
        }

    except json.JSONDecodeError:
        fallback_text = raw if raw else text

        return {
            "original_text":  text,
            "mitigated_text": fallback_text,
            "final_text":     fallback_text,

            "changes":        [],

            # ✅ ADD THESE
            "rule_changes":   [],
            "spacy_changes":  [],
            "bert_changes":   [],
            "cda_changes":    [],
            "wordnet_changes":[],
            "svo_changes":    [],

            "stage1_text": fallback_text,
            "stage2_text": fallback_text,
            "stage3_text": fallback_text,
            "stage4_text": fallback_text,
            "stage5_text": fallback_text,
            "stage6_text": fallback_text,

            "summary":        "Mitigation applied but changelog could not be parsed.",
            "bias_reduced":   True,
            "strategy":       "LLM rewrite (partial — JSON parse failed)"
        }

    except Exception as e:
        return {
            "original_text":  text,
            "mitigated_text": text,
            "final_text":     text,

            "changes":        [],

            # ✅ ADD THESE
            "rule_changes":   [],
            "spacy_changes":  [],
            "bert_changes":   [],
            "cda_changes":    [],
            "wordnet_changes":[],
            "svo_changes":    [],

            "stage1_text": text,
            "stage2_text": text,
            "stage3_text": text,
            "stage4_text": text,
            "stage5_text": text,
            "stage6_text": text,

            "summary":        f"Mitigation failed: {e}",
            "bias_reduced":   False,
            "strategy":       "Failed"
        }
