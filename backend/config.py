from pydantic_settings import BaseSettings
from pydantic import field_validator


class Settings(BaseSettings):
    # Database — SQLite for local dev; set DATABASE_URL env var in production
    database_url: str = "sqlite+aiosqlite:///./emotionmirror.db"

    redis_url: str = "redis://localhost:6379"

    secret_key: str = "change-me-in-production"
    algorithm: str = "HS256"
    access_token_expire_minutes: int = 60 * 24 * 7  # 7 days

    model_checkpoint: str = "model/checkpoints/efficientnet_b0_fer2013_best.pt"
    model_onnx: str = "model/checkpoints/efficientnet_b0_fer2013.onnx"
    use_onnx: bool = False

    anthropic_api_key: str = ""
    llm_model: str = "claude-sonnet-4-6"

    # CORS — set ALLOWED_ORIGINS as a JSON array in the environment:
    #   ALLOWED_ORIGINS=["https://your-app.vercel.app"]
    allowed_origins: list[str] = [
        "http://localhost:3000",
        "http://localhost:5173",
        "http://localhost:5174",
        "http://localhost:8081",
    ]

    @field_validator("database_url", mode="before")
    @classmethod
    def fix_postgres_url(cls, v: str) -> str:
        # Render (and Heroku) provide "postgres://" URLs; asyncpg needs "postgresql+asyncpg://"
        if v.startswith("postgres://"):
            return v.replace("postgres://", "postgresql+asyncpg://", 1)
        if v.startswith("postgresql://") and "+asyncpg" not in v:
            return v.replace("postgresql://", "postgresql+asyncpg://", 1)
        return v

    class Config:
        env_file = ".env"


settings = Settings()
