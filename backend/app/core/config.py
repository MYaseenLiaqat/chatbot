import os
from dataclasses import dataclass

from dotenv import load_dotenv

load_dotenv("backend/.env")
load_dotenv(".env")


@dataclass(frozen=True)
class Settings:
    groq_api_key: str | None = os.getenv("GROQ_API_KEY")
    groq_model: str = os.getenv("GROQ_MODEL", "openai/gpt-oss-20b")
    frontend_url: str = os.getenv("FRONTEND_URL", "http://localhost:3000")
    max_upload_size: int = 10 * 1024 * 1024
    max_github_repositories: int = 10
    max_readme_size: int = 100 * 1024


settings = Settings()
