"""Relationship System module for Haimiya Senpai Discord Bot.

Handles per-user affection points, relationship levels, cooldowns,
daily caps, level progression bar, and prompt context injection.
"""

from dataclasses import dataclass
from datetime import datetime, timezone
import math
from pathlib import Path
from typing import Any

from config import (
    RELATIONSHIP_COOLDOWN_SECONDS,
    RELATIONSHIP_DAILY_CAP,
    RELATIONSHIP_POINTS_PER_INTERACTION,
)
from database import (
    admin_add_points,
    admin_reset_relationship,
    admin_set_points,
    get_user_relationship,
    record_interaction,
)


@dataclass(frozen=True)
class LevelDefinition:
    """Definisi satu tingkat relationship."""

    level: int
    min_points: int
    max_points: int | None
    status: str
    title: str
    badge: str
    color: int
    description: str
    prompt_instruction: str


# ---------------------------------------------------------------------------
# Konfigurasi Tingkat Hubungan (Mudah disesuaikan)
# ---------------------------------------------------------------------------
RELATIONSHIP_LEVELS: list[LevelDefinition] = [
    LevelDefinition(
        level=1,
        min_points=0,
        max_points=19,
        status="Acquaintance",
        title="Kenalan Baru",
        badge="🌱",
        color=0x95A5A6,  # Abu-abu / Silver muda
        description="Baru saling mengenal. Haimiya bersikap ramah, sopan, dan menjaga jarak wajar.",
        prompt_instruction=(
            "HUBUNGAN DENGAN USER SAAT INI: [Acquaintance / Kenalan Baru (Level 1)]. "
            "Kamu baru mengenal user ini. Bersikaplah ramah dan sopan selayaknya baru berkenalan. "
            "Jaga batas wajar, jangan terlalu akrab, jangan memanggil dengan julukan khusus, dan jangan berlebihan menggoda."
        ),
    ),
    LevelDefinition(
        level=2,
        min_points=20,
        max_points=59,
        status="Familiar",
        title="Mulai Akrab",
        badge="🌿",
        color=0x2ECC71,  # Hijau segar
        description="Mulai saling mengenal. Haimiya lebih santai dan mulai mengenali user.",
        prompt_instruction=(
            "HUBUNGAN DENGAN USER SAAT INI: [Familiar / Mulai Akrab (Level 2)]. "
            "Kamu sudah mulai terbiasa mengobrol dengan user ini. Gaya bicaramu lebih santai, "
            "hangat, dan mulai menunjukkan sapaan yang bersahabat seperti teman satu lingkungan."
        ),
    ),
    LevelDefinition(
        level=3,
        min_points=60,
        max_points=119,
        status="Close Friend",
        title="Teman Dekat",
        badge="⭐",
        color=0xF1C40F,  # Emas
        description="Teman dekat. Haimiya nyaman mengobrol, lebih akrab, dan sering bercanda.",
        prompt_instruction=(
            "HUBUNGAN DENGAN USER SAAT INI: [Close Friend / Teman Dekat (Level 3)]. "
            "User ini adalah teman dekatmu. Kamu merasa akrab, santai, sering bercanda, "
            "suka menimpali obrolan dengan antusias, dan menganggapnya sebagai teman nongkrong yang asyik."
        ),
    ),
    LevelDefinition(
        level=4,
        min_points=120,
        max_points=199,
        status="Bestie",
        title="Sahabat Karib",
        badge="💖",
        color=0xE91E63,  # Merah muda / Magenta
        description="Sahabat karib. Haimiya sangat nyaman, playful, dan suka teasing secara bersahabat.",
        prompt_instruction=(
            "HUBUNGAN DENGAN USER SAAT INI: [Bestie / Sahabat Karib (Level 4)]. "
            "User ini adalah sahabat terbaikmu! Kamu sangat nyaman bercanda spontan, "
            "suka teasing secara lucu dan bersahabat, menunjukkan perhatian tulus, dan tidak canggung sama sekali."
        ),
    ),
    LevelDefinition(
        level=5,
        min_points=200,
        max_points=None,
        status="Special",
        title="Teman Istimewa",
        badge="👑",
        color=0x9B59B6,  # Ungu Royal
        description="Hubungan sangat personal, hangat, dan memiliki rasa saling percaya yang tinggi.",
        prompt_instruction=(
            "HUBUNGAN DENGAN USER SAAT INI: [Special / Teman Istimewa (Level 5)]. "
            "Hubunganmu dengan user ini berada pada tingkat kedekatan tertinggi dan sangat personal. "
            "Kamu sangat peduli, menyambutnya dengan hangat, sangat mempercayainya, dan merasa nyaman membicarakan apa saja. "
            "Tetap pertahankan konsep anime companion yang bersahabat (bukan romantis/bucin yang berlebihan)."
        ),
    ),
]


def get_level_definition(points: int) -> LevelDefinition:
    """Tentukan tingkatan level berdasarkan jumlah affection points."""
    safe_points = max(0, points)
    for lvl in reversed(RELATIONSHIP_LEVELS):
        if safe_points >= lvl.min_points:
            return lvl
    return RELATIONSHIP_LEVELS[0]


def build_progress_bar(points: int, bar_length: int = 10) -> tuple[str, float, int, int]:
    """
    Hitung progress bar visual dan detail angka progress level.

    Returns:
        (progress_bar_str, percentage, current_in_level, needed_in_level)
    """
    safe_points = max(0, points)
    lvl_def = get_level_definition(safe_points)

    if lvl_def.max_points is None:
        # Max Level (Special / Level 5)
        bar = "█" * bar_length
        return f"[{bar}]", 100.0, safe_points - lvl_def.min_points, 0

    total_needed = (lvl_def.max_points - lvl_def.min_points) + 1
    current_in_level = safe_points - lvl_def.min_points
    ratio = min(1.0, max(0.0, current_in_level / total_needed))
    percent = ratio * 100.0

    filled_chars = int(round(ratio * bar_length))
    empty_chars = max(0, bar_length - filled_chars)
    bar = ("█" * filled_chars) + ("░" * empty_chars)

    return f"[{bar}]", percent, current_in_level, total_needed


@dataclass
class UserRelationshipData:
    """Data lengkap status relationship seorang user."""

    guild_id: int
    user_id: int
    affection_points: int
    interaction_count: int
    last_interaction_at: str | None
    daily_points: int
    daily_reset_date: str
    created_at: str
    updated_at: str

    @property
    def level_def(self) -> LevelDefinition:
        return get_level_definition(self.affection_points)

    @property
    def level(self) -> int:
        return self.level_def.level

    @property
    def status(self) -> str:
        return self.level_def.status

    @property
    def title(self) -> str:
        return self.level_def.title

    @property
    def badge(self) -> str:
        return self.level_def.badge

    @property
    def color(self) -> int:
        return self.level_def.color

    @property
    def description(self) -> str:
        return self.level_def.description

    @property
    def prompt_instruction(self) -> str:
        return self.level_def.prompt_instruction

    @property
    def progress(self) -> tuple[str, float, int, int]:
        return build_progress_bar(self.affection_points)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "UserRelationshipData":
        return cls(
            guild_id=data["guild_id"],
            user_id=data["user_id"],
            affection_points=data.get("affection_points", 0),
            interaction_count=data.get("interaction_count", 0),
            last_interaction_at=data.get("last_interaction_at"),
            daily_points=data.get("daily_points", 0),
            daily_reset_date=data.get("daily_reset_date", ""),
            created_at=data.get("created_at", ""),
            updated_at=data.get("updated_at", ""),
        )

    @classmethod
    def default_for(cls, guild_id: int, user_id: int) -> "UserRelationshipData":
        now_iso = datetime.now(timezone.utc).isoformat()
        date_str = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        return cls(
            guild_id=guild_id,
            user_id=user_id,
            affection_points=0,
            interaction_count=0,
            last_interaction_at=None,
            daily_points=0,
            daily_reset_date=date_str,
            created_at=now_iso,
            updated_at=now_iso,
        )


def fetch_user_relationship(
    guild_id: int, user_id: int, db_path: Path | None = None
) -> UserRelationshipData:
    """Ambil data relationship user dari database atau kembalikan profil default level 1."""
    data = get_user_relationship(guild_id, user_id, db_path=db_path)
    if data:
        return UserRelationshipData.from_dict(data)
    return UserRelationshipData.default_for(guild_id, user_id)


def process_interaction_reward(
    guild_id: int,
    user_id: int,
    points_to_add: int = RELATIONSHIP_POINTS_PER_INTERACTION,
    cooldown_seconds: int = RELATIONSHIP_COOLDOWN_SECONDS,
    daily_cap: int = RELATIONSHIP_DAILY_CAP,
    now: datetime | None = None,
    db_path: Path | None = None,
) -> tuple[UserRelationshipData, bool, str]:
    """
    Eksekusi penambahan poin interaksi ke database dengan cooldown dan daily cap.

    Returns:
        (relationship_data, awarded_bool, reason_str)
    """
    row, awarded, reason = record_interaction(
        guild_id=guild_id,
        user_id=user_id,
        points_to_add=points_to_add,
        cooldown_seconds=cooldown_seconds,
        daily_cap=daily_cap,
        now=now,
        db_path=db_path,
    )
    return UserRelationshipData.from_dict(row), awarded, reason


# ---------------------------------------------------------------------------
# Admin API Helpers
# ---------------------------------------------------------------------------
def set_user_points(guild_id: int, user_id: int, points: int) -> UserRelationshipData:
    row = admin_set_points(guild_id, user_id, points)
    return UserRelationshipData.from_dict(row)


def add_user_points(guild_id: int, user_id: int, delta: int) -> UserRelationshipData:
    row = admin_add_points(guild_id, user_id, delta)
    return UserRelationshipData.from_dict(row)


def reset_user_relationship(guild_id: int, user_id: int) -> UserRelationshipData:
    admin_reset_relationship(guild_id, user_id)
    return UserRelationshipData.default_for(guild_id, user_id)
