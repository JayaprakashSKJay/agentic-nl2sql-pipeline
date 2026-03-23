from __future__ import annotations

import json
import re
import tempfile
from typing import Dict, Tuple

from llm_factory import get_llm


def transcribe_and_translate_audio_openai(audio_bytes: bytes) -> Dict[str, str]:
    """
    Voice pipeline for multilingual speech input.

    Returns:
    {
        "original_text": "<transcribed text in detected/original language>",
        "detected_language": "<language code or unknown>",
        "english_text": "<english translation of the same transcription>"
    }
    """
    from openai import OpenAI

    client = OpenAI()

    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
        tmp.write(audio_bytes)
        tmp_path = tmp.name

    # Step 1: transcription in original language
    with open(tmp_path, "rb") as f:
        transcript = client.audio.transcriptions.create(
            model="whisper-1",
            file=f,
            response_format="verbose_json",
        )

    original_text = (getattr(transcript, "text", "") or "").strip()
    detected_language = (getattr(transcript, "language", None) or "unknown").strip()

    # Step 2: English translation
    if original_text:
        with open(tmp_path, "rb") as f:
            translated = client.audio.translations.create(
                model="whisper-1",
                file=f,
            )
        english_text = (getattr(translated, "text", "") or "").strip()
        if not english_text:
            english_text = original_text
    else:
        english_text = ""

    return {
        "original_text": original_text,
        "detected_language": detected_language,
        "english_text": english_text,
    }


def normalize_question_to_english(
    provider_name: str,
    question: str,
    source_language: str,
) -> Tuple[str, str]:
    """
    Translate typed input to English for NL-to-SQL.
    Returns:
      (english_question, detected_language)
    """
    q = (question or "").strip()
    src = (source_language or "auto").strip().lower()

    if not q:
        return "", src

    # If user explicitly says English, skip translation
    if src in {"english", "en"}:
        return q, "english"

    llm = get_llm(provider_name)

    prompt = f"""
You are a translation engine for a text-to-SQL system.

Translate the user's database question into English.
Preserve database meaning exactly.
If the question is already in English, return it unchanged.

Return STRICT JSON only:

{{
  "language": "<detected language>",
  "english_question": "<translated english question>"
}}

Question:
{q}
"""

    out = llm.invoke(prompt)
    content = getattr(out, "content", str(out))

    # Extract JSON safely
    m = re.search(r"\{.*\}", content, re.DOTALL)
    if not m:
        return q, src

    try:
        obj = json.loads(m.group(0))
        english_question = (obj.get("english_question", q) or q).strip()
        detected_language = (obj.get("language", src) or src).strip()
        return english_question, detected_language
    except Exception:
        return q, src