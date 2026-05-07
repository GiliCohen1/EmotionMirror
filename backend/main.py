from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from backend.config import settings
from backend.db.session import create_tables
from backend.routers import auth, predict, sessions, journal
from backend.services.inference import load_model


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    await create_tables()
    load_model(settings.model_checkpoint, settings.model_onnx, settings.use_onnx)
    yield
    # Shutdown (nothing to clean up)


app = FastAPI(
    title="EmotionMirror API",
    description="Real-time facial emotion detection & journaling",
    version="0.1.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.allowed_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(auth.router)
app.include_router(predict.router)
app.include_router(sessions.router)
app.include_router(journal.router)


@app.get("/health")
async def health():
    return {"status": "ok"}
