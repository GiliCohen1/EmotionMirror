"""
/api/journal/prompt — generate a reflective journaling question from a session's emotion data.
"""

import json
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from backend.auth import get_current_user
from backend.db.models import EmotionSession, User
from backend.db.session import get_db
from backend.services.journal import generate_journal_prompt

router = APIRouter(prefix="/api/journal", tags=["journal"])


class PromptRequest(BaseModel):
    session_id: int
    note: Optional[str] = None


class PromptResponse(BaseModel):
    prompt: str
    session_id: int


@router.post("/prompt", response_model=PromptResponse)
async def get_prompt(
    body: PromptRequest,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    result = await db.execute(
        select(EmotionSession)
        .where(EmotionSession.id == body.session_id, EmotionSession.user_id == current_user.id)
        .options(selectinload(EmotionSession.readings))
    )
    session = result.scalar_one_or_none()
    if not session:
        raise HTTPException(404, "Session not found")

    readings = [
        {"emotion": r.emotion, "confidence": r.confidence}
        for r in (session.readings or [])
    ]

    prompt_text = generate_journal_prompt(readings, user_note=body.note or session.note)
    return PromptResponse(prompt=prompt_text, session_id=body.session_id)
