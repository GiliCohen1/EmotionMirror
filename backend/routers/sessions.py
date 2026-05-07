"""
Session management — start/stop emotion sessions, fetch history.
"""

import json
from datetime import datetime, timezone
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from backend.auth import get_current_user
from backend.db.models import EmotionReading, EmotionSession, User
from backend.db.session import get_db

router = APIRouter(prefix="/api/sessions", tags=["sessions"])


class SessionCreate(BaseModel):
    note: Optional[str] = None


class SessionEnd(BaseModel):
    note: Optional[str] = None


class ReadingOut(BaseModel):
    id: int
    emotion: str
    confidence: float
    probabilities: dict[str, float]
    timestamp: datetime

    class Config:
        from_attributes = True


class SessionOut(BaseModel):
    id: int
    started_at: datetime
    ended_at: Optional[datetime]
    note: Optional[str]
    readings: list[ReadingOut] = []

    class Config:
        from_attributes = True


@router.post("", response_model=SessionOut, status_code=201)
async def start_session(
    body: SessionCreate,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    session = EmotionSession(user_id=current_user.id, note=body.note)
    db.add(session)
    await db.commit()
    await db.refresh(session)
    # Don't call _session_to_out here — it would try to lazy-load readings
    # in an async context. New sessions have no readings yet.
    return SessionOut(
        id=session.id,
        started_at=session.started_at,
        ended_at=None,
        note=session.note,
        readings=[],
    )


@router.patch("/{session_id}/end", response_model=SessionOut)
async def end_session(
    session_id: int,
    body: SessionEnd,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    result = await db.execute(
        select(EmotionSession)
        .where(EmotionSession.id == session_id, EmotionSession.user_id == current_user.id)
        .options(selectinload(EmotionSession.readings))
    )
    session = result.scalar_one_or_none()
    if not session:
        raise HTTPException(404, "Session not found")

    session.ended_at = datetime.now(timezone.utc)
    if body.note:
        session.note = body.note
    await db.commit()
    await db.refresh(session, ["readings"])
    return _session_to_out(session)


@router.get("", response_model=list[SessionOut])
async def list_sessions(
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    result = await db.execute(
        select(EmotionSession)
        .where(EmotionSession.user_id == current_user.id)
        .options(selectinload(EmotionSession.readings))
        .order_by(EmotionSession.started_at.desc())
    )
    sessions = result.scalars().all()
    return [_session_to_out(s) for s in sessions]


@router.get("/{session_id}", response_model=SessionOut)
async def get_session(
    session_id: int,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    result = await db.execute(
        select(EmotionSession)
        .where(EmotionSession.id == session_id, EmotionSession.user_id == current_user.id)
        .options(selectinload(EmotionSession.readings))
    )
    session = result.scalar_one_or_none()
    if not session:
        raise HTTPException(404, "Session not found")
    return _session_to_out(session)


def _session_to_out(session: EmotionSession) -> SessionOut:
    readings = []
    for r in (session.readings or []):
        try:
            probs = json.loads(r.all_probabilities)
        except Exception:
            probs = {}
        readings.append(ReadingOut(
            id=r.id,
            emotion=r.emotion,
            confidence=r.confidence,
            probabilities=probs,
            timestamp=r.timestamp,
        ))
    return SessionOut(
        id=session.id,
        started_at=session.started_at,
        ended_at=session.ended_at,
        note=session.note,
        readings=readings,
    )
