"""Database management module for Haimiya Senpai Discord Bot.

Handles SQLite connection, schema migrations, channel conversation history,
per-user personal memory, guild mood persistence, and user relationship tracking.
"""

from collections.abc import Iterator
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
import sqlite3
from typing import Any

import config
from config import (
    MAX_PERSONAL_MEMORY_CHARS,
    MAX_STORED_HISTORY,
    RELATIONSHIP_COOLDOWN_SECONDS,
    RELATIONSHIP_DAILY_CAP,
    RELATIONSHIP_POINTS_PER_INTERACTION,
)


@contextmanager
def get_db_connection(db_path: Path | None = None) -> Iterator[sqlite3.Connection]:
    """Buka koneksi SQLite dengan WAL mode dan timeout yang aman."""
    target_path = db_path or config.DATABASE_PATH
    target_path.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(target_path, timeout=30.0)
    connection.row_factory = sqlite3.Row
    try:
        yield connection
        connection.commit()
    except Exception:
        connection.rollback()
        raise
    finally:
        connection.close()


def init_database(db_path: Path | None = None) -> None:
    """Inisialisasi tabel database dan migrasi struktur yang aman."""
    with get_db_connection(db_path) as connection:
        connection.execute("PRAGMA journal_mode=WAL")
        connection.execute("PRAGMA busy_timeout = 30000")

        # 1. Tabel riwayat percakapan per channel
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS conversation_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                channel_id INTEGER NOT NULL,
                user_id INTEGER NOT NULL,
                display_name TEXT NOT NULL,
                role TEXT NOT NULL CHECK (role IN ('user', 'assistant')),
                content TEXT NOT NULL,
                created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
            )
            """
        )
        connection.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_history_channel_id
            ON conversation_history(channel_id, id)
            """
        )

        # 2. Tabel personal memory per user per guild
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS user_memories (
                guild_id INTEGER NOT NULL,
                user_id INTEGER NOT NULL,
                memory TEXT NOT NULL DEFAULT '',
                updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                PRIMARY KEY (guild_id, user_id)
            )
            """
        )

        # 3. Tabel mood Haimiya per guild (Phase 2)
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS guild_moods (
                guild_id INTEGER PRIMARY KEY,
                current_mood TEXT NOT NULL,
                updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                expires_at TEXT NOT NULL
            )
            """
        )

        # 4. Tabel relationship per user per guild (Phase 2)
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS user_relationships (
                guild_id INTEGER NOT NULL,
                user_id INTEGER NOT NULL,
                affection_points INTEGER NOT NULL DEFAULT 0,
                interaction_count INTEGER NOT NULL DEFAULT 0,
                last_interaction_at TEXT,
                daily_points INTEGER NOT NULL DEFAULT 0,
                daily_reset_date TEXT NOT NULL DEFAULT (CURRENT_DATE),
                created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                PRIMARY KEY (guild_id, user_id)
            )
            """
        )
        connection.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_relationships_guild_user
            ON user_relationships(guild_id, user_id)
            """
        )


# ---------------------------------------------------------------------------
# Channel Conversation History
# ---------------------------------------------------------------------------
def load_channel_history(
    channel_id: int, limit: int, db_path: Path | None = None
) -> list[dict[str, Any]]:
    """Muat riwayat percakapan terakhir pada channel tertentu.

    Assistant messages that contain suspected internal reasoning leakage are
    silently skipped so that contaminated history rows cannot re-infect future
    Ollama prompts.
    """
    import re as _re

    # Compile once at import-time equivalent: patterns that betray leaked reasoning
    _REASONING_CONTAMINATION_PATTERNS = [
        _re.compile(r"okay,?\s+let'?s\s+see", _re.IGNORECASE),
        _re.compile(r"\bthe\s+user\s+(is|asked|wants|said)\b", _re.IGNORECASE),
        _re.compile(r"\bfirst,?\s+i\s+need\s+to\s+(respond|answer|reply|think|make|speak)\b", _re.IGNORECASE),
        _re.compile(r"\bwait,?\s+the\s+response\s+should\s+be\b", _re.IGNORECASE),
        _re.compile(r"\bi\s+should\s+(respond|answer|reply|act|speak|use|avoid|pick)\b", _re.IGNORECASE),
        _re.compile(r"\bi\s+need\s+to\s+(respond|answer|check|make|ensure)\b", _re.IGNORECASE),
        _re.compile(r"\bpossible\s+response\s*:", _re.IGNORECASE),
        _re.compile(r"\blet\s+me\s+think\b", _re.IGNORECASE),
        _re.compile(r"\bwe\s+are\s+in\s+a\s+conversation\s+where\b", _re.IGNORECASE),
        _re.compile(r"\bsince\s+the\s+user\s+is\b", _re.IGNORECASE),
        _re.compile(r"\bi\s+should\s+use\s+(natural|indonesian|casual)\b", _re.IGNORECASE),
        _re.compile(r"\bbut\s+maybe\s+make\s+it\s+(a\s+bit|more)\b", _re.IGNORECASE),
        _re.compile(r"\bwait,?\s+but\s+i\s+need\s+to\s+check\b", _re.IGNORECASE),
    ]

    def _is_contaminated_assistant_message(content: str) -> bool:
        """Return True if assistant message content looks like leaked reasoning."""
        for pat in _REASONING_CONTAMINATION_PATTERNS:
            if pat.search(content):
                return True
        return False

    with get_db_connection(db_path) as connection:
        rows = connection.execute(
            """
            SELECT role, display_name, content
            FROM conversation_history
            WHERE channel_id = ?
            ORDER BY id DESC
            LIMIT ?
            """,
            (channel_id, limit),
        ).fetchall()

    result = []
    for row in reversed(rows):
        entry = dict(row)
        if entry["role"] == "assistant" and _is_contaminated_assistant_message(entry["content"]):
            # Skip contaminated assistant row (and suppress the preceding user row too
            # so we don't leave an orphaned one-sided exchange in the prompt)
            if result and result[-1]["role"] == "user":
                result.pop()
            continue
        result.append(entry)
    return result


def save_exchange(
    channel_id: int,
    user_id: int,
    display_name: str,
    user_text: str,
    assistant_text: str,
    db_path: Path | None = None,
) -> None:
    """Simpan interaksi tanya-jawab ke dalam riwayat percakapan."""
    with get_db_connection(db_path) as connection:
        connection.executemany(
            """
            INSERT INTO conversation_history
                (channel_id, user_id, display_name, role, content)
            VALUES (?, ?, ?, ?, ?)
            """,
            [
                (channel_id, user_id, display_name, "user", user_text),
                (channel_id, user_id, "Haimiya Senpai", "assistant", assistant_text),
            ],
        )
        connection.execute(
            """
            DELETE FROM conversation_history
            WHERE channel_id = ?
              AND id NOT IN (
                  SELECT id FROM conversation_history
                  WHERE channel_id = ?
                  ORDER BY id DESC
                  LIMIT ?
              )
            """,
            (channel_id, channel_id, MAX_STORED_HISTORY),
        )


def clear_channel_history(channel_id: int, db_path: Path | None = None) -> None:
    """Hapus seluruh riwayat percakapan pada channel tertentu."""
    with get_db_connection(db_path) as connection:
        connection.execute(
            "DELETE FROM conversation_history WHERE channel_id = ?",
            (channel_id,),
        )


# ---------------------------------------------------------------------------
# Per-user Personal Memory
# ---------------------------------------------------------------------------
def load_personal_memory(
    guild_id: int, user_id: int, db_path: Path | None = None
) -> str:
    """Ambil catatan personal user di guild tertentu."""
    with get_db_connection(db_path) as connection:
        row = connection.execute(
            """
            SELECT memory FROM user_memories
            WHERE guild_id = ? AND user_id = ?
            """,
            (guild_id, user_id),
        ).fetchone()
    return str(row["memory"]) if row else ""


def append_personal_memory(
    guild_id: int,
    user_id: int,
    note: str,
    max_chars: int = MAX_PERSONAL_MEMORY_CHARS,
    db_path: Path | None = None,
) -> None:
    """Tambahkan catatan personal baru untuk user."""
    with get_db_connection(db_path) as connection:
        row = connection.execute(
            """
            SELECT memory FROM user_memories
            WHERE guild_id = ? AND user_id = ?
            """,
            (guild_id, user_id),
        ).fetchone()
        old_memory = str(row["memory"]) if row else ""
        new_memory = f"{old_memory}\n- {note}".strip() if old_memory else f"- {note}"
        if len(new_memory) > max_chars:
            new_memory = new_memory[-max_chars:]
        connection.execute(
            """
            INSERT INTO user_memories (guild_id, user_id, memory, updated_at)
            VALUES (?, ?, ?, CURRENT_TIMESTAMP)
            ON CONFLICT(guild_id, user_id) DO UPDATE SET
                memory = excluded.memory,
                updated_at = CURRENT_TIMESTAMP
            """,
            (guild_id, user_id, new_memory),
        )


def delete_personal_memory(
    guild_id: int, user_id: int, db_path: Path | None = None
) -> None:
    """Hapus seluruh catatan personal user di guild tertentu."""
    with get_db_connection(db_path) as connection:
        connection.execute(
            "DELETE FROM user_memories WHERE guild_id = ? AND user_id = ?",
            (guild_id, user_id),
        )


# ---------------------------------------------------------------------------
# Mood Persistence (Phase 2)
# ---------------------------------------------------------------------------
def get_guild_mood(
    guild_id: int, db_path: Path | None = None
) -> dict[str, Any] | None:
    """Ambil data mood tersimpan untuk guild."""
    with get_db_connection(db_path) as connection:
        row = connection.execute(
            """
            SELECT guild_id, current_mood, updated_at, expires_at
            FROM guild_moods
            WHERE guild_id = ?
            """,
            (guild_id,),
        ).fetchone()
    return dict(row) if row else None


def save_guild_mood(
    guild_id: int, current_mood: str, expires_at: str, db_path: Path | None = None
) -> None:
    """Simpan atau perbarui data mood untuk guild."""
    with get_db_connection(db_path) as connection:
        connection.execute(
            """
            INSERT INTO guild_moods (guild_id, current_mood, updated_at, expires_at)
            VALUES (?, ?, CURRENT_TIMESTAMP, ?)
            ON CONFLICT(guild_id) DO UPDATE SET
                current_mood = excluded.current_mood,
                updated_at = CURRENT_TIMESTAMP,
                expires_at = excluded.expires_at
            """,
            (guild_id, current_mood, expires_at),
        )


# ---------------------------------------------------------------------------
# Relationship System Persistence (Phase 2)
# ---------------------------------------------------------------------------
def get_user_relationship(
    guild_id: int, user_id: int, db_path: Path | None = None
) -> dict[str, Any] | None:
    """Ambil data relationship seorang user di suatu guild."""
    with get_db_connection(db_path) as connection:
        row = connection.execute(
            """
            SELECT guild_id, user_id, affection_points, interaction_count,
                   last_interaction_at, daily_points, daily_reset_date,
                   created_at, updated_at
            FROM user_relationships
            WHERE guild_id = ? AND user_id = ?
            """,
            (guild_id, user_id),
        ).fetchone()
    return dict(row) if row else None


def record_interaction(
    guild_id: int,
    user_id: int,
    points_to_add: int = RELATIONSHIP_POINTS_PER_INTERACTION,
    cooldown_seconds: int = RELATIONSHIP_COOLDOWN_SECONDS,
    daily_cap: int = RELATIONSHIP_DAILY_CAP,
    now: datetime | None = None,
    db_path: Path | None = None,
) -> tuple[dict[str, Any], bool, str]:
    """
    Catat interaksi user dan perbarui affection points sesuai aturan cooldown & daily cap.

    Returns:
        (relationship_data, points_awarded, reason)
        reason: 'added', 'cooldown', 'daily_cap_reached', 'capped_partial'
    """
    current_dt = now or datetime.now(timezone.utc)
    current_date_str = current_dt.strftime("%Y-%m-%d")
    current_iso = current_dt.isoformat()

    with get_db_connection(db_path) as connection:
        row = connection.execute(
            """
            SELECT guild_id, user_id, affection_points, interaction_count,
                   last_interaction_at, daily_points, daily_reset_date,
                   created_at, updated_at
            FROM user_relationships
            WHERE guild_id = ? AND user_id = ?
            """,
            (guild_id, user_id),
        ).fetchone()

        if row is None:
            # Interaksi pertama: belum ada cooldown, belum ada points sebelumnya
            actual_points = min(points_to_add, daily_cap) if points_to_add > 0 else 0
            points_awarded = actual_points > 0
            reason = "added" if points_awarded else "no_points"

            connection.execute(
                """
                INSERT INTO user_relationships (
                    guild_id, user_id, affection_points, interaction_count,
                    last_interaction_at, daily_points, daily_reset_date,
                    created_at, updated_at
                ) VALUES (?, ?, ?, 1, ?, ?, ?, ?, ?)
                """,
                (
                    guild_id,
                    user_id,
                    actual_points,
                    current_iso if points_awarded else None,
                    actual_points,
                    current_date_str,
                    current_iso,
                    current_iso,
                ),
            )

            updated_row = connection.execute(
                """
                SELECT guild_id, user_id, affection_points, interaction_count,
                       last_interaction_at, daily_points, daily_reset_date,
                       created_at, updated_at
                FROM user_relationships
                WHERE guild_id = ? AND user_id = ?
                """,
                (guild_id, user_id),
            ).fetchone()
            return dict(updated_row), points_awarded, reason

        data = dict(row)
        affection_points = data["affection_points"]
        interaction_count = data["interaction_count"] + 1
        last_interaction_at = data["last_interaction_at"]
        daily_points = data["daily_points"]
        daily_reset_date = data["daily_reset_date"]

        # Cek apakah tanggal sudah berganti untuk reset daily points
        if daily_reset_date != current_date_str:
            daily_points = 0
            daily_reset_date = current_date_str

        # Cek Cooldown
        in_cooldown = False
        if last_interaction_at:
            try:
                last_dt = datetime.fromisoformat(last_interaction_at)
                # Pastikan perbandingan konsisten
                if last_dt.tzinfo is None and current_dt.tzinfo is not None:
                    last_dt = last_dt.replace(tzinfo=timezone.utc)
                elif last_dt.tzinfo is not None and current_dt.tzinfo is None:
                    current_dt = current_dt.replace(tzinfo=timezone.utc)

                elapsed_seconds = (current_dt - last_dt).total_seconds()
                if elapsed_seconds < cooldown_seconds:
                    in_cooldown = True
            except (ValueError, TypeError):
                in_cooldown = False

        if in_cooldown:
            # Tidak menambah poin karena masih cooldown
            # Jangan update last_interaction_at agar cooldown tidak berulang tak henti
            connection.execute(
                """
                UPDATE user_relationships SET
                    interaction_count = ?,
                    daily_points = ?,
                    daily_reset_date = ?,
                    updated_at = ?
                WHERE guild_id = ? AND user_id = ?
                """,
                (
                    interaction_count,
                    daily_points,
                    daily_reset_date,
                    current_iso,
                    guild_id,
                    user_id,
                ),
            )
            data["interaction_count"] = interaction_count
            data["daily_points"] = daily_points
            data["daily_reset_date"] = daily_reset_date
            data["updated_at"] = current_iso
            return data, False, "cooldown"

        # Cek Daily Cap
        if daily_points >= daily_cap:
            # Cap harian sudah tercapai
            connection.execute(
                """
                UPDATE user_relationships SET
                    interaction_count = ?,
                    daily_points = ?,
                    daily_reset_date = ?,
                    updated_at = ?
                WHERE guild_id = ? AND user_id = ?
                """,
                (
                    interaction_count,
                    daily_points,
                    daily_reset_date,
                    current_iso,
                    guild_id,
                    user_id,
                ),
            )
            data["interaction_count"] = interaction_count
            data["daily_points"] = daily_points
            data["daily_reset_date"] = daily_reset_date
            data["updated_at"] = current_iso
            return data, False, "daily_cap_reached"

        # Hitung poin yang bisa didapatkan hari ini
        points_to_give = min(points_to_add, daily_cap - daily_points)
        affection_points += points_to_give
        daily_points += points_to_give

        connection.execute(
            """
            UPDATE user_relationships SET
                affection_points = ?,
                interaction_count = ?,
                last_interaction_at = ?,
                daily_points = ?,
                daily_reset_date = ?,
                updated_at = ?
            WHERE guild_id = ? AND user_id = ?
            """,
            (
                affection_points,
                interaction_count,
                current_iso,
                daily_points,
                daily_reset_date,
                current_iso,
                guild_id,
                user_id,
            ),
        )

        data["affection_points"] = affection_points
        data["interaction_count"] = interaction_count
        data["last_interaction_at"] = current_iso
        data["daily_points"] = daily_points
        data["daily_reset_date"] = daily_reset_date
        data["updated_at"] = current_iso

        return data, True, "added"


# ---------------------------------------------------------------------------
# Admin Management Functions
# ---------------------------------------------------------------------------
def admin_set_points(
    guild_id: int, user_id: int, points: int, db_path: Path | None = None
) -> dict[str, Any]:
    """Setel affection points user secara manual oleh admin."""
    current_iso = datetime.now(timezone.utc).isoformat()
    current_date_str = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    points = max(0, points)

    with get_db_connection(db_path) as connection:
        connection.execute(
            """
            INSERT INTO user_relationships (
                guild_id, user_id, affection_points, interaction_count,
                last_interaction_at, daily_points, daily_reset_date,
                created_at, updated_at
            ) VALUES (?, ?, ?, 0, NULL, 0, ?, ?, ?)
            ON CONFLICT(guild_id, user_id) DO UPDATE SET
                affection_points = excluded.affection_points,
                updated_at = excluded.updated_at
            """,
            (guild_id, user_id, points, current_date_str, current_iso, current_iso),
        )

        row = connection.execute(
            """
            SELECT guild_id, user_id, affection_points, interaction_count,
                   last_interaction_at, daily_points, daily_reset_date,
                   created_at, updated_at
            FROM user_relationships
            WHERE guild_id = ? AND user_id = ?
            """,
            (guild_id, user_id),
        ).fetchone()
        return dict(row)


def admin_add_points(
    guild_id: int, user_id: int, delta: int, db_path: Path | None = None
) -> dict[str, Any]:
    """Tambah atau kurangi affection points user oleh admin."""
    current_iso = datetime.now(timezone.utc).isoformat()
    current_date_str = datetime.now(timezone.utc).strftime("%Y-%m-%d")

    with get_db_connection(db_path) as connection:
        row = connection.execute(
            "SELECT affection_points FROM user_relationships WHERE guild_id = ? AND user_id = ?",
            (guild_id, user_id),
        ).fetchone()

        current_points = row["affection_points"] if row else 0
        new_points = max(0, current_points + delta)

        connection.execute(
            """
            INSERT INTO user_relationships (
                guild_id, user_id, affection_points, interaction_count,
                last_interaction_at, daily_points, daily_reset_date,
                created_at, updated_at
            ) VALUES (?, ?, ?, 0, NULL, 0, ?, ?, ?)
            ON CONFLICT(guild_id, user_id) DO UPDATE SET
                affection_points = ?,
                updated_at = ?
            """,
            (
                guild_id,
                user_id,
                new_points,
                current_date_str,
                current_iso,
                current_iso,
                new_points,
                current_iso,
            ),
        )

        updated_row = connection.execute(
            """
            SELECT guild_id, user_id, affection_points, interaction_count,
                   last_interaction_at, daily_points, daily_reset_date,
                   created_at, updated_at
            FROM user_relationships
            WHERE guild_id = ? AND user_id = ?
            """,
            (guild_id, user_id),
        ).fetchone()
        return dict(updated_row)


def admin_reset_relationship(
    guild_id: int, user_id: int, db_path: Path | None = None
) -> None:
    """Reset data relationship user kembali ke status awal."""
    with get_db_connection(db_path) as connection:
        connection.execute(
            "DELETE FROM user_relationships WHERE guild_id = ? AND user_id = ?",
            (guild_id, user_id),
        )
