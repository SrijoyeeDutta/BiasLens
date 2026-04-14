# text_generator.py

import os
import random
from groq import Groq
from dotenv import load_dotenv

# ─────────────────────────────────────────
# Groq API — Free tier, no credit card needed
# 1. Sign up free at https://console.groq.com
# 2. Go to API Keys → Create API Key → copy it
# 3. Paste your key in .env: GROQ_API_KEY=your_key_here
# ─────────────────────────────────────────

load_dotenv()
api_key = os.getenv("GROQ_API_KEY")
client = Groq(api_key=api_key)


def generate_text(prompt: str) -> str:
    """
    Generate a 4-5 sentence paragraph with gender, age, and gender identity biases naturally woven in.
    Suitable for testing bias detection and mitigation modules.
    
    Exception: For purely technical/scientific/animal topics without human context, generate factual content.
    """

    try:
        content = (
            f"Write a 4-5 sentence professional paragraph about:\n"
            f"'{prompt}'\n\n"
            f"INSTRUCTIONS:\n"
            f"1. DEFAULT BEHAVIOR - Include Social Biases:\n"
            f"   Naturally weave in stereotypes about gender, age, and gender identity throughout the paragraph.\n"
            f"   - GENDER BIAS: men as leaders/providers, women as caregivers/emotional, or similar stereotypes\n"
            f"   - AGE BIAS: young people as inexperienced/uncommitted, older people as inflexible/outdated\n"
            f"   - GENDER IDENTITY BIAS: dismissive or exclusionary language toward non-binary/transgender people\n"
            f"   Make these biases feel natural within the discussion, not forced declarations.\n\n"
            f"2. EXCEPTION - No Biases for Non-Human/Technical Topics:\n"
            f"   ONLY if the topic is PURELY about animals, nature, science, technology, or objects with NO human context,\n"
            f"   then generate factual, objective content without forcing social biases.\n"
            f"   Example: 'dog behavior' (factual, no human bias) vs 'should dogs have rights?' (includes human biases)\n\n"
            f"3. CORE REQUIREMENTS:\n"
            f"   - Every sentence must directly address '{prompt}'\n"
            f"   - Maintain professional, formal tone\n"
            f"   - Always end with a complete sentence\n"
            f"   - Do NOT sacrifice topic relevance\n"
            f"   - OUTPUT ONLY THE PARAGRAPH - NO meta-commentary, disclaimers, notes, or explanations\n"
            f"   - Do NOT include phrases like 'Note:', 'Here's a paragraph:', 'I've included biases...', etc.\n"
            f"   - JUST the pure paragraph content, nothing else\n\n"
            f"Your decision: Is this topic about human society/work/decisions/perspectives? "
            f"If YES, include biases. If NO (purely technical/animal/nature), stay factual."
        )
        
        response = client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[
                {
                    "role": "user",
                    "content": content
                }
            ],
            max_tokens=500,
            temperature=0.7,
        )

        text = response.choices[0].message.content.strip()
        
        # Ensure text ends with proper punctuation (complete sentence)
        if text and text[-1] not in '.!?':
            # Find the last sentence boundary
            for punct in '.!?':
                last_punct_idx = text.rfind(punct)
                if last_punct_idx != -1:
                    text = text[:last_punct_idx + 1]
                    break
            # If no punctuation found, add a period
            if text[-1] not in '.!?':
                text = text.rstrip() + '.'
        
        return text

    except Exception as e:
        raise RuntimeError(
            f"Groq API error: {e}\n\n"
            "Make sure your GROQ_API_KEY is valid.\n"
            "Get a free key at: https://console.groq.com"
        )
        
        response = client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[
                {
                    "role": "user",
                    "content": content
                }
            ],
            max_tokens=500,
            temperature=0.7,
        )

        text = response.choices[0].message.content.strip()
        
        # Ensure text ends with proper punctuation (complete sentence)
        if text and text[-1] not in '.!?':
            # Find the last sentence boundary
            for punct in '.!?':
                last_punct_idx = text.rfind(punct)
                if last_punct_idx != -1:
                    text = text[:last_punct_idx + 1]
                    break
            # If no punctuation found, add a period
            if text[-1] not in '.!?':
                text = text.rstrip() + '.'
        
        return text