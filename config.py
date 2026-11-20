"""Configuration module for Haimiya Senpai Discord Bot."""

import os
from pathlib import Path
from dotenv import load_dotenv

# Muat environment variables dari .env
load_dotenv()

# Discord & Core Settings
DISCORD_TOKEN = os.getenv("DISCORD_TOKEN")
DATABASE_PATH = Path(os.getenv("DATABASE_PATH", "data/haimiya.db"))

# Ollama LLM Settings
OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://localhost:11434")
MODEL = os.getenv("OLLAMA_MODEL", "qwen3.5:4b")
NUM_CTX = int(os.getenv("OLLAMA_NUM_CTX", "4096"))
MAX_TOKENS = int(os.getenv("OLLAMA_MAX_TOKENS", "600"))
# Default: thinking mode dimatikan (False) agar Qwen3 langsung menghasilkan respons final
THINK_MODE = os.getenv("OLLAMA_THINK", "false").strip().lower()
OLLAMA_THINK = THINK_MODE in ("true", "1", "yes")  # Boolean False secara default

# Conversation History & Memory Settings
MAX_HISTORY = max(2, int(os.getenv("MAX_HISTORY", "12")))
MAX_STORED_HISTORY = max(100, MAX_HISTORY * 10)
MAX_PERSONAL_MEMORY_CHARS = 1500

# ---------------------------------------------------------------------------
# Phase 2: Mood System Configuration
# ---------------------------------------------------------------------------
MOOD_MIN_DURATION_MINUTES = int(os.getenv("MOOD_MIN_DURATION_MINUTES", "30"))
MOOD_MAX_DURATION_MINUTES = int(os.getenv("MOOD_MAX_DURATION_MINUTES", "60"))

# ---------------------------------------------------------------------------
# Phase 2: Relationship System Configuration
# ---------------------------------------------------------------------------
RELATIONSHIP_POINTS_PER_INTERACTION = int(os.getenv("RELATIONSHIP_POINTS_PER_INTERACTION", "1"))
RELATIONSHIP_COOLDOWN_SECONDS = int(os.getenv("RELATIONSHIP_COOLDOWN_SECONDS", "60"))
RELATIONSHIP_DAILY_CAP = int(os.getenv("RELATIONSHIP_DAILY_CAP", "20"))
