from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    # Database — SQLite by default for local dev, swap to postgresql+asyncpg for prod
    database_url: str = "sqlite+aiosqlite:///./emotionmirror.db"

    # Redis (optional — not used yet in local dev)
    redis_url: str = "redis://localhost:6379"

    # Auth
    secret_key: str = "change-me-in-production"
    algorithm: str = "HS256"
    access_token_expire_minutes: int = 60 * 24 * 7  # 7 days

    # Model
    model_checkpoint: str = "model/checkpoints/efficientnet_b0_fer2013_best.pt"
    model_onnx: str = "model/checkpoints/efficientnet_b0_fer2013.onnx"
    use_onnx: bool = False  # flip to True after exporting ONNX

    # LLM
    anthropic_api_key: str = ""
    llm_model: str = "claude-sonnet-4-6"

    # CORS
    allowed_origins: list[str] = ["http://localhost:3000", "http://localhost:5173", "http://localhost:5174", "http://localhost:8081"]

    class Config:
        env_file = ".env"


settings = Settings()
