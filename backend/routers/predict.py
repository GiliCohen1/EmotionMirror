"""
/api/predict  — single-image REST endpoint
/ws/stream    — WebSocket for real-time webcam frames
"""

import asyncio
import json
import logging
from typing import Optional

from fastapi import APIRouter, Depends, WebSocket, WebSocketDisconnect
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from backend.auth import get_current_user
from backend.db.models import EmotionReading, EmotionSession, User
from backend.db.session import get_db, AsyncSessionLocal
from backend.services.inference import predict_from_base64

logger = logging.getLogger(__name__)


async def _run_inference(image_b64: str) -> dict:
    """Run synchronous inference in a thread pool so it doesn't block the event loop."""
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(None, predict_from_base64, image_b64)

router = APIRouter(prefix="/api", tags=["predict"])


class PredictRequest(BaseModel):
    image: str   # base64-encoded frame (data URI or raw base64)
    session_id: Optional[int] = None


class PredictResponse(BaseModel):
    emotion: Optional[str]
    confidence: float
    probabilities: dict[str, float]
    face_found: bool


@router.post("/predict", response_model=PredictResponse)
async def predict(
    body: PredictRequest,
    current_user: User = Depends(get_current_user),
    db=Depends(get_db),
):
    try:
        result = await _run_inference(body.image)
    except Exception as exc:
        logger.exception("Inference error")
        return JSONResponse(status_code=500, content={"detail": str(exc)})

    if result.get("face_found") and body.session_id:
        reading = EmotionReading(
            session_id=body.session_id,
            emotion=result["emotion"],
            confidence=result["confidence"],
            all_probabilities=json.dumps(result["probabilities"]),
        )
        db.add(reading)
        await db.commit()

    return PredictResponse(
        emotion=result.get("emotion"),
        confidence=result.get("confidence", 0.0),
        probabilities=result.get("probabilities", {}),
        face_found=result.get("face_found", False),
    )


@router.websocket("/ws/stream")
async def stream(websocket: WebSocket):
    """
    WebSocket endpoint for real-time emotion detection.

    Client sends: { "image": "<base64>", "session_id": <int|null> }
    Server sends: { "emotion": "happy", "confidence": 0.92, "probabilities": {...}, "face_found": true }

    No auth on WebSocket — caller must pass token as query param: /ws/stream?token=<jwt>
    """
    token = websocket.query_params.get("token")
    await websocket.accept()

    # Validate token
    if not token:
        await websocket.send_json({"error": "Missing token"})
        await websocket.close(code=4001)
        return

    from backend.auth import get_current_user
    from fastapi.security import OAuth2PasswordBearer
    from backend.db.session import AsyncSessionLocal
    from backend.db.models import User
    from sqlalchemy import select
    from jose import JWTError, jwt
    from backend.config import settings

    try:
        payload = jwt.decode(token, settings.secret_key, algorithms=[settings.algorithm])
        user_id = int(payload["sub"])
    except (JWTError, KeyError, ValueError):
        await websocket.send_json({"error": "Invalid token"})
        await websocket.close(code=4001)
        return

    try:
        while True:
            data = await websocket.receive_json()
            image_b64 = data.get("image", "")
            session_id = data.get("session_id")

            if not image_b64:
                await websocket.send_json({"error": "Missing image"})
                continue

            result = await _run_inference(image_b64)

            # Persist to DB if session active
            if result.get("face_found") and session_id:
                async with AsyncSessionLocal() as db:
                    reading = EmotionReading(
                        session_id=session_id,
                        emotion=result["emotion"],
                        confidence=result["confidence"],
                        all_probabilities=json.dumps(result["probabilities"]),
                    )
                    db.add(reading)
                    await db.commit()

            await websocket.send_json(result)

    except WebSocketDisconnect:
        pass
