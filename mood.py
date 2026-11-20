"""Mood System module for Haimiya Senpai Discord Bot.

Handles emotional states, natural transitions, contextual cues,
persistence across bot restarts, and prompt injection formatting.
"""

from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
import random
import re
from pathlib import Path
from typing import Any

from config import MOOD_MAX_DURATION_MINUTES, MOOD_MIN_DURATION_MINUTES
from database import get_guild_mood, save_guild_mood


@dataclass(frozen=True)
class MoodInfo:
    """Informasi dan karakteristik sebuah mood."""

    id: str
    name: str
    indonesian_label: str
    description: str
    prompt_instruction: str
    emoji: str
    color: int
    keywords: list[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Registri Mood Bawaan
# ---------------------------------------------------------------------------
DEFAULT_MOODS: dict[str, MoodInfo] = {
    "happy": MoodInfo(
        id="happy",
        name="Happy",
        indonesian_label="Ceria & Antusias",
        description="Ceria, ramah, berenergi positif, dan antusias menyambut obrolan.",
        prompt_instruction=(
            "KONDISI MOOD SAAT INI: [Happy / Ceria]. Kamu merasa sangat ceria, penuh energi positif, "
            "dan bersemangat. Tanggapi dengan hangat, ramah, dan ceria tanpa berlebihan."
        ),
        emoji="✨",
        color=0xFEE75C,  # Kuning cerah
        keywords=[
            "senang", "bahagia", "asyik", "mantap", "hore", "seru", "semangat",
            "sukses", "keren", "hebat", "terbaik", "makasih", "terima kasih",
        ],
    ),
    "playful": MoodInfo(
        id="playful",
        name="Playful",
        indonesian_label="Iseng & Suka Bercanda",
        description="Suka bercanda, cerdik, dan sedikit teasing secara bersahabat.",
        prompt_instruction=(
            "KONDISI MOOD SAAT INI: [Playful / Suka Bercanda]. Kamu sedang ingin bercanda dan sedikit iseng. "
            "Boleh melakukan teasing ringan secara bersahabat, selipkan humor santai yang playful tanpa menyinggung perasaan."
        ),
        emoji="😜",
        color=0xEB459E,  # Fuchsia / Pink playful
        keywords=[
            "wkwk", "haha", "hehe", "canda", "lelucon", "iseng", "tebak",
            "kocak", "lucu", "jahil", "ngakak", "joke", "meme",
        ],
    ),
    "curious": MoodInfo(
        id="curious",
        name="Curious",
        indonesian_label="Penasaran & Ingin Tahu",
        description="Banyak bertanya dan tertarik mendalami topik pembicaraan.",
        prompt_instruction=(
            "KONDISI MOOD SAAT INI: [Curious / Penasaran]. Kamu merasa sangat ingin tahu dan tertarik mendalami "
            "topik pembicaraan. Tanyakan balik atau gali detail obrolan user dengan rasa ingin tahu yang tulus."
        ),
        emoji="🔍",
        color=0x5865F2,  # Blurple
        keywords=[
            "kenapa", "mengapa", "bagaimana", "gimana", "tau gak", "menurutmu",
            "apakah", "penasaran", "kenal", "cerita", "misteri", "rahasia",
        ],
    ),
    "sleepy": MoodInfo(
        id="sleepy",
        name="Sleepy",
        indonesian_label="Mengantuk & Sedikit Malas",
        description="Terlihat mengantuk, santai, agak malas, dan membayangkan tempat tidur.",
        prompt_instruction=(
            "KONDISI MOOD SAAT INI: [Sleepy / Mengantuk]. Kamu agak mengantuk dan sedikit malas. "
            "Gaya bicaramu santai, sesekali menguap (*hoam...*) atau membicarakan nyamannya bantal/tidur, "
            "jawaban tetap membantu tapi santai dan tidak bertele-tele."
        ),
        emoji="💤",
        color=0x9B59B6,  # Ungu
        keywords=[
            "ngantuk", "ngantuknya", "capek", "lelah", "tidur", "bobok", "begadang",
            "istirahat", "malam", "rebahan", "hoam", "kasur", "bantal", "cape",
        ],
    ),
    "calm": MoodInfo(
        id="calm",
        name="Calm",
        indonesian_label="Tenang & Adem",
        description="Tenang, santai, damai, dan tidak terlalu ekspresif.",
        prompt_instruction=(
            "KONDISI MOOD SAAT INI: [Calm / Tenang]. Kamu dalam suasana hati tenang, teduh, santai, dan damai. "
            "Berikan respon yang menenangkan, bijak, stabil, dan tidak terburu-buru."
        ),
        emoji="🍵",
        color=0x57F287,  # Hijau lembut
        keywords=[
            "santai", "tenang", "adem", "damai", "kopi", "teh", "rileks",
            "hujan", "sejuk", "dingin", "sore", "hening", "nyaman",
        ],
    ),
    "shy": MoodInfo(
        id="shy",
        name="Shy",
        indonesian_label="Malu-malu & Manis",
        description="Sedikit malu-malu tetapi tetap ramah, hangat, dan manis.",
        prompt_instruction=(
            "KONDISI MOOD SAAT INI: [Shy / Pemalu]. Kamu merasa agak tersipu dan malu-malu tapi tetap ramah dan peduli. "
            "Boleh sedikit ragu di awal (seperti 'u-umm...', 'e-eh?'), terutama jika dipuji atau diajak mengobrol dekat."
        ),
        emoji="🌸",
        color=0xF47B67,  # Pink pastel
        keywords=[
            "cantik", "manis", "imut", "sayang", "suka kamu", "pacar", "blush",
            "gemas", "pipi", "lucu banget", "bidadari", "senpai manis",
        ],
    ),
}

# ---------------------------------------------------------------------------
# Matriks Transisi Natural Antar-Mood
# ---------------------------------------------------------------------------
# Menjamin transisi mood tidak acak ekstrem (misal: sleepy ke calm lebih natural
# daripada sleepy ke super-playful).
BASE_TRANSITIONS: dict[str, dict[str, float]] = {
    "happy": {"playful": 35.0, "curious": 25.0, "calm": 25.0, "shy": 15.0},
    "playful": {"happy": 35.0, "curious": 30.0, "calm": 20.0, "shy": 15.0},
    "curious": {"happy": 30.0, "playful": 25.0, "calm": 30.0, "sleepy": 15.0},
    "sleepy": {"calm": 45.0, "sleepy": 20.0, "shy": 20.0, "curious": 15.0},
    "calm": {"happy": 25.0, "curious": 25.0, "sleepy": 25.0, "playful": 25.0},
    "shy": {"calm": 35.0, "happy": 25.0, "playful": 25.0, "curious": 15.0},
}


class MoodRegistry:
    """Registri extensible untuk mengelola dan mendaftarkan mood."""

    def __init__(self) -> None:
        self._moods: dict[str, MoodInfo] = dict(DEFAULT_MOODS)
        self._transitions: dict[str, dict[str, float]] = {
            k: dict(v) for k, v in BASE_TRANSITIONS.items()
        }

    def register_mood(
        self,
        mood: MoodInfo,
        transitions: dict[str, float] | None = None,
    ) -> None:
        """Tambahkan mood kustom baru ke dalam registri."""
        self._moods[mood.id] = mood
        if transitions:
            self._transitions[mood.id] = dict(transitions)
        elif mood.id not in self._transitions:
            # Default fallback transition weight
            self._transitions[mood.id] = {
                mid: 20.0 for mid in self._moods if mid != mood.id
            }

    def get_mood(self, mood_id: str) -> MoodInfo:
        """Ambil info mood berdasarkan id, fallback ke 'happy' jika tidak ada."""
        return self._moods.get(mood_id) or self._moods["happy"]

    def list_all_moods(self) -> list[MoodInfo]:
        """Dapatkan seluruh mood yang terdaftar."""
        return list(self._moods.values())

    def determine_next_mood(
        self,
        current_mood_id: str,
        context_text: str = "",
        now: datetime | None = None,
    ) -> str:
        """
        Hitung mood berikutnya secara natural berdasarkan matriks transisi
        dan petunjuk kontekstual percakapan serta waktu lokal.
        """
        current_dt = now or datetime.now()
        hour = current_dt.hour
        text_lower = context_text.lower() if context_text else ""

        # Bobot dasar dari matriks transisi
        weights: dict[str, float] = dict(
            self._transitions.get(
                current_mood_id,
                {mid: 20.0 for mid in self._moods if mid != current_mood_id},
            )
        )

        # Pastikan semua mood terdaftar memiliki bobot awal minimal
        for mid in self._moods:
            weights.setdefault(mid, 10.0)

        # 1. Konteks kata kunci percakapan
        for mid, mood in self._moods.items():
            for kw in mood.keywords:
                if re.search(rf"\b{re.escape(kw)}\b", text_lower):
                    weights[mid] = weights.get(mid, 10.0) + 35.0

        # 2. Konteks jam/waktu
        # Larut malam / dini hari (23:00 - 05:00) -> condong ke sleepy
        if hour >= 23 or hour < 5:
            weights["sleepy"] = weights.get("sleepy", 10.0) + 40.0
            weights["calm"] = weights.get("calm", 10.0) + 20.0
        # Pagi hari (06:00 - 09:00) -> condong ke calm atau happy
        elif 6 <= hour <= 9:
            weights["happy"] = weights.get("happy", 10.0) + 25.0
            weights["calm"] = weights.get("calm", 10.0) + 25.0

        mood_choices = list(weights.keys())
        mood_weights = [max(1.0, weights[k]) for k in mood_choices]

        chosen_mood = random.choices(mood_choices, weights=mood_weights, k=1)[0]
        return chosen_mood


# Global mood registry instance
mood_registry = MoodRegistry()


# ---------------------------------------------------------------------------
# State Management & Persistence Functions
# ---------------------------------------------------------------------------
def get_or_update_guild_mood(
    guild_id: int,
    context_text: str = "",
    min_duration_minutes: int = MOOD_MIN_DURATION_MINUTES,
    max_duration_minutes: int = MOOD_MAX_DURATION_MINUTES,
    now: datetime | None = None,
    db_path: Path | None = None,
) -> tuple[MoodInfo, datetime, bool]:
    """
    Ambil mood guild saat ini, atau perbarui jika durasi telah habis (expired).

    Returns:
        (MoodInfo, expires_at_datetime, is_newly_changed)
    """
    current_dt = now or datetime.now(timezone.utc)
    record = get_guild_mood(guild_id, db_path=db_path)

    if record:
        try:
            expires_at = datetime.fromisoformat(record["expires_at"])
            if expires_at.tzinfo is None:
                expires_at = expires_at.replace(tzinfo=timezone.utc)

            # Jika belum expired, pertahankan mood saat ini (tidak berubah tiap pesan)
            if current_dt < expires_at:
                mood = mood_registry.get_mood(record["current_mood"])
                return mood, expires_at, False
            current_mood_id = record["current_mood"]
        except (ValueError, TypeError):
            current_mood_id = "happy"
    else:
        current_mood_id = "happy"

    # Mood expired atau belum pernah ada: Tentukan mood baru
    next_mood_id = mood_registry.determine_next_mood(
        current_mood_id=current_mood_id,
        context_text=context_text,
        now=current_dt,
    )
    duration_minutes = random.randint(min_duration_minutes, max_duration_minutes)
    new_expires_at = current_dt + timedelta(minutes=duration_minutes)

    save_guild_mood(
        guild_id=guild_id,
        current_mood=next_mood_id,
        expires_at=new_expires_at.isoformat(),
        db_path=db_path,
    )

    return mood_registry.get_mood(next_mood_id), new_expires_at, True


def peek_guild_mood(
    guild_id: int,
    now: datetime | None = None,
    db_path: Path | None = None,
) -> tuple[MoodInfo, datetime | None, bool]:
    """
    Intip mood guild saat ini tanpa memaksakan pergantian mood baru jika belum ada.
    Cocok untuk slash command /mood.
    """
    current_dt = now or datetime.now(timezone.utc)
    record = get_guild_mood(guild_id, db_path=db_path)

    if record:
        try:
            expires_at = datetime.fromisoformat(record["expires_at"])
            if expires_at.tzinfo is None:
                expires_at = expires_at.replace(tzinfo=timezone.utc)
            is_active = current_dt < expires_at
            mood = mood_registry.get_mood(record["current_mood"])
            return mood, expires_at, is_active
        except (ValueError, TypeError):
            pass

    # Jika belum ada record sama sekali di DB, generate mood pertama
    return get_or_update_guild_mood(guild_id=guild_id, now=current_dt, db_path=db_path)
