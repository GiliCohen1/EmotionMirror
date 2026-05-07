"""
LLM journaling service: generates reflective prompts based on the user's
emotion history using the Claude API.
"""

from collections import Counter
from typing import Optional

import anthropic

from backend.config import settings

EMOTIONS = ["angry", "disgust", "fear", "happy", "neutral", "sad", "surprise"]


def _summarize_session(readings: list[dict]) -> str:
    if not readings:
        return "No emotion data recorded."

    counts = Counter(r["emotion"] for r in readings if r.get("emotion"))
    total = sum(counts.values())
    dominant = counts.most_common(3)

    lines = [f"Session had {total} readings."]
    lines.append("Dominant emotions:")
    for emotion, count in dominant:
        pct = round(100 * count / total)
        lines.append(f"  - {emotion}: {pct}%")
    return "\n".join(lines)


def generate_journal_prompt(
    readings: list[dict],
    user_note: Optional[str] = None,
) -> str:
    """
    Calls Claude to generate a reflective journaling question based on
    the emotion pattern detected during the session.
    """
    if not settings.anthropic_api_key:
        return "What patterns do you notice in how you felt during this session, and what might have influenced them?"

    client = anthropic.Anthropic(api_key=settings.anthropic_api_key)

    session_summary = _summarize_session(readings)

    user_content = f"""I just finished an emotion-tracking session.

{session_summary}
"""
    if user_note:
        user_content += f'\nMy note about this session: "{user_note}"'

    user_content += """

Based on this emotional pattern, give me one thoughtful journaling question that helps me reflect
on what I was experiencing. The question should be specific to my emotional pattern — not generic.
Reply with the question only, no preamble."""

    message = client.messages.create(
        model=settings.llm_model,
        max_tokens=150,
        system=(
            "You are a compassionate mindfulness coach who helps people reflect on their emotions. "
            "You have a background in cognitive science and evidence-based therapy. "
            "Your questions are warm, specific, and grounded in the user's actual experience."
        ),
        messages=[{"role": "user", "content": user_content}],
    )
    return message.content[0].text.strip()
