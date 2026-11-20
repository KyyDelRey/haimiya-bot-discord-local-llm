"""Unit and Integration Tests for Haimiya Senpai Phase 2 & Anti-Reasoning Bugfix.

Covers:
1. Python syntax and module imports
2. Database initialization and safe schema migration
3. Relationship points calculation
4. Cooldown enforcement
5. Daily cap enforcement and day-reset
6. Relationship level progression (Levels 1-5, progress bars)
7. Mood persistence, duration, and context-aware transitions
8. Context injection into System Prompt with Anti-Reasoning directives
9. Regression check for existing Phase 1 features (history, memory, clean_answer, split_message)
10. Ollama ask_ai mock flow & error handling
11. Explicit think=False boolean parameter enforcement
12. Strict exclusion of message.thinking field
13. Cleaning of leaked thought preambles while preserving legitimate technical answers
14. Natural fallback handling for empty final responses
"""

import sys
from pathlib import Path

# Tambahkan root directory ke sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# ---------------------------------------------------------------------------
# Sandbox compatibility patch: Windows Sandbox network isolation blocks
# localhost socketpair, so we use native PostQueuedCompletionStatus for IOCP wakeup.
# ---------------------------------------------------------------------------
if sys.platform == "win32":
    import _overlapped
    from asyncio import proactor_events, windows_events

    class _MockLoopSocket:
        def fileno(self):
            return -1

        def setblocking(self, b):
            pass

        def close(self):
            pass

    proactor_events.BaseProactorEventLoop._make_self_pipe = lambda self: (
        setattr(self, "_ssock", _MockLoopSocket()),
        setattr(self, "_csock", _MockLoopSocket()),
        setattr(self, "_internal_fds", self._internal_fds + 1),
    )
    proactor_events.BaseProactorEventLoop._loop_self_reading = lambda self, f=None: None
    proactor_events.BaseProactorEventLoop._write_to_self = (
        lambda self: _overlapped.PostQueuedCompletionStatus(self._proactor._iocp, 0, 0, 0)
        if hasattr(self, "_proactor") and hasattr(self._proactor, "_iocp")
        else None
    )

    _orig_poll = windows_events.IocpProactor._poll

    def _safe_poll(self, timeout=None):
        try:
            return _orig_poll(self, timeout)
        except (TimeoutError, OSError) as e:
            if getattr(e, "winerror", None) in (0, 258) or isinstance(e, TimeoutError):
                return
            raise

    windows_events.IocpProactor._poll = _safe_poll

import asyncio
from datetime import datetime, timedelta, timezone
import sqlite3
import tempfile
import unittest
from unittest.mock import AsyncMock, MagicMock, patch

from config import (
    RELATIONSHIP_COOLDOWN_SECONDS,
    RELATIONSHIP_DAILY_CAP,
    RELATIONSHIP_POINTS_PER_INTERACTION,
)
import database
import mood
import relationship
import main


class TestPhase2Haimiya(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.test_db_path = Path(self.temp_dir.name) / "test_haimiya.db"
        database.init_database(self.test_db_path)

    def tearDown(self) -> None:
        self.temp_dir.cleanup()

    # -----------------------------------------------------------------------
    # Test 1 & 2: Database Initialization & Safe Schema Migration
    # -----------------------------------------------------------------------
    def test_database_initialization_and_schema(self) -> None:
        """Pastikan tabel lama dan tabel baru Phase 2 terbuat dengan benar."""
        with database.get_db_connection(self.test_db_path) as conn:
            tables = [
                row[0]
                for row in conn.execute(
                    "SELECT name FROM sqlite_master WHERE type='table'"
                ).fetchall()
            ]

        expected_tables = {
            "conversation_history",
            "user_memories",
            "guild_moods",
            "user_relationships",
        }
        for table in expected_tables:
            self.assertIn(table, tables, f"Tabel {table} harus ada di database.")

    def test_database_preserves_existing_data(self) -> None:
        """Pastikan pemanggilan ulang init_database tidak menghapus data yang sudah ada."""
        database.save_exchange(
            channel_id=101,
            user_id=202,
            display_name="Tester",
            user_text="Halo Haimiya!",
            assistant_text="Halo juga Tester!",
            db_path=self.test_db_path,
        )
        database.append_personal_memory(
            guild_id=303,
            user_id=202,
            note="Suka matcha latte",
            db_path=self.test_db_path,
        )

        # Jalankan init ulang
        database.init_database(self.test_db_path)

        history = database.load_channel_history(
            channel_id=101, limit=10, db_path=self.test_db_path
        )
        self.assertEqual(len(history), 2)
        memory = database.load_personal_memory(
            guild_id=303, user_id=202, db_path=self.test_db_path
        )
        self.assertIn("Suka matcha latte", memory)

    # -----------------------------------------------------------------------
    # Test 3 & 4: Relationship Points Calculation & Cooldown
    # -----------------------------------------------------------------------
    def test_relationship_points_calculation_and_cooldown(self) -> None:
        """Uji penambahan affection point dan pembatasan cooldown."""
        t0 = datetime(2026, 11, 11, 10, 0, 0, tzinfo=timezone.utc)
        guild_id, user_id = 1001, 2001

        # Interaksi pertama -> Poin masuk
        data1, awarded1, reason1 = database.record_interaction(
            guild_id=guild_id,
            user_id=user_id,
            points_to_add=1,
            cooldown_seconds=60,
            daily_cap=20,
            now=t0,
            db_path=self.test_db_path,
        )
        self.assertTrue(awarded1)
        self.assertEqual(reason1, "added")
        self.assertEqual(data1["affection_points"], 1)
        self.assertEqual(data1["interaction_count"], 1)

        # Interaksi kedua 20 detik kemudian -> Cooldown aktif, tidak dapat poin
        t1 = t0 + timedelta(seconds=20)
        data2, awarded2, reason2 = database.record_interaction(
            guild_id=guild_id,
            user_id=user_id,
            points_to_add=1,
            cooldown_seconds=60,
            daily_cap=20,
            now=t1,
            db_path=self.test_db_path,
        )
        self.assertFalse(awarded2)
        self.assertEqual(reason2, "cooldown")
        self.assertEqual(data2["affection_points"], 1)
        self.assertEqual(data2["interaction_count"], 2)

        # Interaksi ketiga 65 detik setelah t0 -> Cooldown selesai, poin bertambah
        t2 = t0 + timedelta(seconds=65)
        data3, awarded3, reason3 = database.record_interaction(
            guild_id=guild_id,
            user_id=user_id,
            points_to_add=1,
            cooldown_seconds=60,
            daily_cap=20,
            now=t2,
            db_path=self.test_db_path,
        )
        self.assertTrue(awarded3)
        self.assertEqual(reason3, "added")
        self.assertEqual(data3["affection_points"], 2)
        self.assertEqual(data3["interaction_count"], 3)

    # -----------------------------------------------------------------------
    # Test 5: Daily Cap & Daily Reset
    # -----------------------------------------------------------------------
    def test_relationship_daily_cap_and_reset(self) -> None:
        """Uji pembatasan daily cap dan reset harian otomatis."""
        t0 = datetime(2026, 11, 11, 8, 0, 0, tzinfo=timezone.utc)
        guild_id, user_id = 1002, 2002
        daily_cap = 5

        # Simulasikan 5 interaksi bertahap di luar cooldown
        for i in range(5):
            t_curr = t0 + timedelta(minutes=i * 2)
            data, awarded, reason = database.record_interaction(
                guild_id=guild_id,
                user_id=user_id,
                points_to_add=1,
                cooldown_seconds=60,
                daily_cap=daily_cap,
                now=t_curr,
                db_path=self.test_db_path,
            )
            self.assertTrue(awarded, f"Interaksi ke-{i+1} harus dapat poin.")
            self.assertEqual(data["daily_points"], i + 1)

        # Interaksi ke-6 di hari yang sama -> Daily cap tercapai
        t_cap = t0 + timedelta(minutes=20)
        data_cap, awarded_cap, reason_cap = database.record_interaction(
            guild_id=guild_id,
            user_id=user_id,
            points_to_add=1,
            cooldown_seconds=60,
            daily_cap=daily_cap,
            now=t_cap,
            db_path=self.test_db_path,
        )
        self.assertFalse(awarded_cap)
        self.assertEqual(reason_cap, "daily_cap_reached")
        self.assertEqual(data_cap["affection_points"], 5)
        self.assertEqual(data_cap["daily_points"], 5)

        # Hari berikutnya (2026-11-12) -> Daily points otomatis ter-reset ke 0
        t_next_day = datetime(2026, 11, 12, 9, 0, 0, tzinfo=timezone.utc)
        data_next, awarded_next, reason_next = database.record_interaction(
            guild_id=guild_id,
            user_id=user_id,
            points_to_add=1,
            cooldown_seconds=60,
            daily_cap=daily_cap,
            now=t_next_day,
            db_path=self.test_db_path,
        )
        self.assertTrue(awarded_next)
        self.assertEqual(reason_next, "added")
        self.assertEqual(data_next["affection_points"], 6)
        self.assertEqual(data_next["daily_points"], 1)

    # -----------------------------------------------------------------------
    # Test 6: Relationship Level Progression & Progress Bar
    # -----------------------------------------------------------------------
    def test_relationship_levels_and_progress_bar(self) -> None:
        """Uji klasifikasi level (1 sampai 5) dan visualisasi progress bar."""
        test_cases = [
            (0, 1, "Acquaintance"),
            (19, 1, "Acquaintance"),
            (20, 2, "Familiar"),
            (59, 2, "Familiar"),
            (60, 3, "Close Friend"),
            (119, 3, "Close Friend"),
            (120, 4, "Bestie"),
            (199, 4, "Bestie"),
            (200, 5, "Special"),
            (350, 5, "Special"),
        ]

        for points, expected_lvl, expected_status in test_cases:
            lvl_def = relationship.get_level_definition(points)
            self.assertEqual(
                lvl_def.level,
                expected_lvl,
                f"Points {points} harus level {expected_lvl}",
            )
            self.assertEqual(
                lvl_def.status,
                expected_status,
                f"Points {points} harus berstatus {expected_status}",
            )

        bar, percent, curr, needed = relationship.build_progress_bar(10)
        self.assertEqual(needed, 20)
        self.assertEqual(curr, 10)
        self.assertEqual(percent, 50.0)
        self.assertIn("█", bar)
        self.assertIn("░", bar)

        bar_max, percent_max, curr_max, needed_max = relationship.build_progress_bar(250)
        self.assertEqual(percent_max, 100.0)
        self.assertEqual(needed_max, 0)
        self.assertEqual(bar_max, "[██████████]")

    # -----------------------------------------------------------------------
    # Test 7: Admin Relationship Controls
    # -----------------------------------------------------------------------
    def test_admin_relationship_controls(self) -> None:
        """Uji fungsi administratif set, add, dan reset relationship."""
        guild_id, user_id = 9001, 8001

        database.admin_set_points(guild_id, user_id, 150, db_path=self.test_db_path)
        row = database.get_user_relationship(guild_id, user_id, db_path=self.test_db_path)
        self.assertIsNotNone(row)
        self.assertEqual(row["affection_points"], 150)
        user_rel = relationship.UserRelationshipData.from_dict(row)
        self.assertEqual(user_rel.level, 4)
        self.assertEqual(user_rel.status, "Bestie")

        database.admin_add_points(guild_id, user_id, 60, db_path=self.test_db_path)
        row = database.get_user_relationship(guild_id, user_id, db_path=self.test_db_path)
        self.assertEqual(row["affection_points"], 210)
        user_rel = relationship.UserRelationshipData.from_dict(row)
        self.assertEqual(user_rel.level, 5)
        self.assertEqual(user_rel.status, "Special")

        database.admin_reset_relationship(guild_id, user_id, db_path=self.test_db_path)
        row = database.get_user_relationship(guild_id, user_id, db_path=self.test_db_path)
        self.assertIsNone(row)

    # -----------------------------------------------------------------------
    # Test 8: Mood Persistence, Duration & Context Transitions
    # -----------------------------------------------------------------------
    def test_mood_persistence_and_duration(self) -> None:
        """Uji persistensi mood, pemulihan setelah restart, dan ketahanan durasi."""
        guild_id = 777
        t0 = datetime(2026, 11, 11, 12, 0, 0, tzinfo=timezone.utc)

        mood1, exp1, changed1 = mood.get_or_update_guild_mood(
            guild_id=guild_id,
            min_duration_minutes=30,
            max_duration_minutes=60,
            now=t0,
            db_path=self.test_db_path,
        )
        self.assertTrue(changed1)
        self.assertIn(mood1.id, mood.DEFAULT_MOODS)

        t1 = t0 + timedelta(minutes=10)
        mood2, exp2, changed2 = mood.get_or_update_guild_mood(
            guild_id=guild_id,
            context_text="halo haimiya!",
            now=t1,
            db_path=self.test_db_path,
        )
        self.assertFalse(changed2)
        self.assertEqual(mood1.id, mood2.id)
        self.assertEqual(exp1, exp2)

        db_record = database.get_guild_mood(guild_id, self.test_db_path)
        self.assertIsNotNone(db_record)
        self.assertEqual(db_record["current_mood"], mood1.id)

        t_expired = t0 + timedelta(minutes=90)
        mood3, exp3, changed3 = mood.get_or_update_guild_mood(
            guild_id=guild_id,
            context_text="aku ngantuk banget mau tidur capek",
            now=t_expired,
            db_path=self.test_db_path,
        )
        self.assertTrue(changed3)
        self.assertGreater(exp3, t_expired)

    def test_mood_context_bias(self) -> None:
        """Uji bahwa mood context bias tidak error dan mengembalikan mood yang valid."""
        registry = mood.MoodRegistry()
        night_time = datetime(2026, 11, 11, 23, 30, 0)
        chosen = registry.determine_next_mood(
            current_mood_id="calm",
            context_text="capek banget seharian, ngantuk pengen tidur rebahan",
            now=night_time,
        )
        # Mood yang dipilih harus selalu merupakan mood yang terdaftar (valid)
        self.assertIn(chosen, mood.DEFAULT_MOODS,
                      f"determine_next_mood harus mengembalikan mood ID yang valid, got: {chosen!r}")

    # -----------------------------------------------------------------------
    # Test 9: Context Injection in System Prompt & Anti-Reasoning Directives
    # -----------------------------------------------------------------------
    def test_system_prompt_context_injection_and_directives(self) -> None:
        """Uji pembangunan prompt lengkap dan pastikan direktif anti-reasoning ada."""
        curious_mood = mood.DEFAULT_MOODS["curious"]
        rel_data = relationship.UserRelationshipData(
            guild_id=1,
            user_id=2,
            affection_points=140,
            interaction_count=45,
            last_interaction_at=None,
            daily_points=5,
            daily_reset_date="2026-11-11",
            created_at="",
            updated_at="",
        )
        personal_memory = "Menyukai game Wuthering Waves dan kopi latte."

        prompt = main.build_system_prompt(
            user_name="Kim Minji",
            personal_memory=personal_memory,
            mood_info=curious_mood,
            relationship=rel_data,
        )

        self.assertIn("Haimiya Senpai", prompt)
        self.assertIn("USER SAAT INI: Kim Minji", prompt)
        self.assertIn("KONDISI SUASANA HATI (MOOD):", prompt)
        self.assertIn("Curious", prompt)
        self.assertIn("TINGKAT KEDEKATAN DENGAN USER (💖 Bestie - Level 4):", prompt)
        self.assertIn("Wuthering Waves", prompt)

        # Direktif Anti-Reasoning Wajib
        self.assertIn("Respond directly to the user", prompt)
        self.assertIn("NEVER reveal internal reasoning", prompt)
        self.assertIn("Do not explain what you are thinking", prompt)
        self.assertIn("Never narrate the user's question back to yourself", prompt)
        self.assertIn("Produce only the final response intended for the user", prompt)
        self.assertIn("Use natural conversational Indonesian when the user speaks Indonesian", prompt)
        self.assertIn("Keep casual conversation responses concise, generally one to three sentences", prompt)
        self.assertIn("Stay in character as Haimiya Senpai", prompt)

    # -----------------------------------------------------------------------
    # Test 10: Regression Check for Existing Utilities
    # -----------------------------------------------------------------------
    def test_clean_answer_and_split_message(self) -> None:
        """Uji fungsi pembersih tag <think> dan pemecah pesan."""
        raw_think = "<think>Analisis pertanyaan user...</think>Halo! Senang mengobrol denganmu."
        cleaned = main.clean_answer(raw_think)
        self.assertEqual(cleaned, "Halo! Senang mengobrol denganmu.")

        normal = "Ini jawaban biasa."
        self.assertEqual(main.clean_answer(normal), normal)

        long_text = "Baris 1\n" * 300
        chunks = main.split_message(long_text, limit=100)
        self.assertTrue(len(chunks) > 1)
        for chunk in chunks:
            self.assertLessEqual(len(chunk), 100)

    # -----------------------------------------------------------------------
    # Test 11: Ask AI Flow with Explicit think=False Parameter
    # -----------------------------------------------------------------------
    def test_ask_ai_think_false_parameter_and_reward(self) -> None:
        """Uji alur ask_ai memastikan think=False diteruskan sebagai boolean eksplisit."""
        async def run_test():
            mock_response = AsyncMock()
            mock_response.message.content = "Halo juga! Senang bisa ngobrol."
            mock_response.message.thinking = None
            mock_response.prompt_eval_count = 120
            mock_response.eval_count = 25

            mock_chat = AsyncMock(return_value=mock_response)
            with patch.object(main.ollama, "chat", mock_chat):
                answer = await main.ask_ai(
                    channel_id=555,
                    guild_id=777,
                    user_id=888,
                    user_name="Budi",
                    text="Hai Haimiya!",
                    db_path=self.test_db_path,
                )
                self.assertEqual(answer, "Halo juga! Senang bisa ngobrol.")

                # Pastikan parameter think=False (boolean murni) diterima oleh ollama.chat
                call_kwargs = mock_chat.call_args.kwargs
                self.assertIn("think", call_kwargs)
                self.assertIs(call_kwargs["think"], False)
                self.assertIsInstance(call_kwargs["think"], bool)

                # Pastikan exchange tersimpan di history
                history = database.load_channel_history(555, 10, db_path=self.test_db_path)
                self.assertEqual(len(history), 2)
                self.assertEqual(history[0]["content"], "Hai Haimiya!")
                self.assertEqual(history[1]["content"], "Halo juga! Senang bisa ngobrol.")

                # Pastikan affection points bertambah ke user
                rel = database.get_user_relationship(777, 888, db_path=self.test_db_path)
                self.assertIsNotNone(rel)
                self.assertEqual(rel["affection_points"], 1)

        asyncio.run(run_test())

    def test_ask_ai_error_does_not_award_points(self) -> None:
        """Uji bahwa jika AI error, tidak ada affection points yang diberikan."""
        async def run_error_test():
            with patch.object(main.ollama, "chat", AsyncMock(side_effect=ConnectionError("Ollama down"))):
                with self.assertRaises(ConnectionError):
                    await main.ask_ai(
                        channel_id=555,
                        guild_id=777,
                        user_id=999,
                        user_name="Andi",
                        text="Halo?",
                        db_path=self.test_db_path,
                    )

                rel = database.get_user_relationship(777, 999, db_path=self.test_db_path)
                self.assertIsNone(rel)

        asyncio.run(run_error_test())

    # -----------------------------------------------------------------------
    # Test 12: Exclusion of message.thinking & Leaked Reasoning Cleaning
    # -----------------------------------------------------------------------
    def test_separate_thinking_field_is_never_sent_or_saved(self) -> None:
        """Pastikan field message.thinking terpisah tidak pernah dikirim ke Discord atau history."""
        async def run_test():
            mock_response = AsyncMock()
            # Simulasi output Ollama terbaru di mana internal reasoning disimpan di message.thinking
            mock_response.message.thinking = "Internal thought process: User is asking about food. I should answer nicely."
            mock_response.message.content = "Belum nih, Haimiya lagi sibuk nemenin kalian wkwk. Kamu sendiri udah makan belum?"
            mock_response.prompt_eval_count = 130
            mock_response.eval_count = 35

            with patch.object(main.ollama, "chat", AsyncMock(return_value=mock_response)):
                answer = await main.ask_ai(
                    channel_id=111,
                    guild_id=222,
                    user_id=333,
                    user_name="Kim Minji",
                    text="kamu udah makan?",
                    db_path=self.test_db_path,
                )

                # Jawaban tidak boleh memuat reasoning
                self.assertNotIn("Internal thought process", answer)
                self.assertNotIn("User is asking", answer)
                self.assertEqual(
                    answer,
                    "Belum nih, Haimiya lagi sibuk nemenin kalian wkwk. Kamu sendiri udah makan belum?",
                )

                # Pastikan yang disimpan ke database riwayat juga hanya jawaban final yang bersih
                history = database.load_channel_history(111, 10, db_path=self.test_db_path)
                self.assertNotIn("Internal thought process", history[1]["content"])

        asyncio.run(run_test())

    def test_leaked_reasoning_preamble_is_cleaned(self) -> None:
        """Uji bahwa contoh kasus riil problem pengguna dibersihkan secara akurat (expanded)."""
        leaked_response = (
            "Okay, let's see. The user is Kim Minji, and she asked, 'kamu udah makan?' which means 'Have you eaten yet?' in Indonesian.\n\n"
            "First, I need to respond as Haimiya Senpai...\n\n"
            "I should use natural Indonesian slang but not too much.\n\n"
            "Possible response: Belum nih, Haimiya lagi sibuk nemenin kalian wkwk.\n\n"
            "Wait, but I need to check if it's too much.\n\n"
            "But maybe make it a bit more playful.\n\n"
            "Wait, the response should be in Indonesian, natural. Let me think...\n\n"
            "Belum nih, Haimiya lagi sibuk nemenin kalian wkwk. Kamu sendiri udah makan belum? Jangan sampai telat makan ya!"
        )

        cleaned = main.clean_answer(leaked_response)

        # Internal monologue reasoning harus lenyap sepenuhnya
        self.assertNotIn("Okay, let's see", cleaned)
        self.assertNotIn("The user is Kim Minji", cleaned)
        self.assertNotIn("First, I need to respond", cleaned)
        self.assertNotIn("I should use natural Indonesian slang", cleaned)
        self.assertNotIn("Wait, but I need to check", cleaned)
        self.assertNotIn("But maybe make it a bit more playful", cleaned)
        self.assertNotIn("Wait, the response should be", cleaned)
        self.assertNotIn("Let me think", cleaned)

        # Jawaban final harus ada dan bersih
        self.assertIn(
            "Belum nih, Haimiya lagi sibuk nemenin kalian wkwk. Kamu sendiri udah makan belum? Jangan sampai telat makan ya!",
            cleaned,
        )

    def test_legitimate_technical_explanations_preserved(self) -> None:
        """Pastikan penjelasan teknis yang mengandung istilah Inggris TIDAK terpotong."""
        tech_cases = [
            (
                "GPU (Graphics Processing Unit) bekerja dengan cara memproses ribuan instruksi secara paralel sekaligus melalui core-core kecil yang disebut streaming multiprocessors. Berbeda dengan CPU yang berfokus pada sequential tasks.",
                "Jelaskan secara detail bagaimana GPU bekerja."
            ),
            (
                "Langit berwarna biru karena fenomena yang disebut hamburan Rayleigh (Rayleigh scattering). Cahaya matahari yang masuk ke atmosfer dihamburkan oleh molekul gas, dan cahaya biru memiliki panjang gelombang lebih pendek sehingga paling banyak dihamburkan.",
                "Kenapa langit berwarna biru?"
            ),
            (
                "Kalau rekomendasi anime romance yang manis dan santai, coba tonton Horimiya atau Kaguya-sama: Love Is War! Dijamin bikin senyum-senyum sendiri wkwk.",
                "Rekomendasi anime romance dong."
            ),
        ]

        for content, question in tech_cases:
            cleaned = main.clean_answer(content)
            self.assertEqual(cleaned, content, f"Jawaban untuk '{question}' harus dipertahankan utuh.")

    def test_empty_content_triggers_friendly_fallback(self) -> None:
        """Uji jika respons model kosong atau hanya berisi tag think, bot memakai fallback alami."""
        async def run_empty_test():
            mock_response = AsyncMock()
            # Model hanya menghasilkan tag think tanpa konten final
            mock_response.message.content = "<think>Thinking intensely about everything...</think>"
            mock_response.message.thinking = None

            with patch.object(main.ollama, "chat", AsyncMock(return_value=mock_response)):
                answer = await main.ask_ai(
                    channel_id=111,
                    guild_id=222,
                    user_id=444,
                    user_name="Budi",
                    text="Halo?",
                    db_path=self.test_db_path,
                )
                self.assertEqual(
                    answer,
                    "Aduh, Haimiya lagi agak bengong nih wkwk. Coba tanya sekali lagi ya!",
                )

        asyncio.run(run_empty_test())


# ---------------------------------------------------------------------------
# Second Regression Test Class — Tests A–F from bug report
# ---------------------------------------------------------------------------
class TestReasoningLeakRegression(unittest.TestCase):
    """
    Regression tests covering the second-investigation bug requirements.
    Tests A through F from the bug report specification.
    """

    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.test_db_path = Path(self.temp_dir.name) / "test_regression.db"
        database.init_database(self.test_db_path)

    def tearDown(self) -> None:
        self.temp_dir.cleanup()

    # -----------------------------------------------------------------------
    # Test A — Reasoning inside unstructured content (expanded patterns)
    # -----------------------------------------------------------------------
    def test_A_unstructured_reasoning_fully_stripped(self) -> None:
        """Test A: Semua pola reasoning bocor dalam konten tidak terstruktur harus dibersihkan."""
        cases = [
            (
                "Okay, let's see. The user is Kim Minji...\n\nHai Kim Minji!",
                "Hai Kim Minji!",
            ),
            (
                "First, I need to respond as Haimiya Senpai...\n\nHai, ada yang mau diobrolkan?",
                "Hai, ada yang mau diobrolkan?",
            ),
            (
                "I should use natural Indonesian slang but not too much.\n\nHalo! Lagi ngapain nih?",
                "Halo! Lagi ngapain nih?",
            ),
            (
                "Possible response: Belum nih, sibuk nemenin kamu wkwk.",
                "Belum nih, sibuk nemenin kamu wkwk.",
            ),
            (
                "Wait, but I need to check if it's too much.\n\nSantai aja yuk!",
                "Santai aja yuk!",
            ),
        ]
        for raw, expected in cases:
            cleaned = main.clean_answer(raw)
            self.assertEqual(cleaned, expected, f"Failed for input: {raw!r}")

    def test_A_mid_response_reasoning_removed(self) -> None:
        """Test A: Reasoning di tengah respons juga harus dibersihkan."""
        raw = (
            "Halo Kim Minji!\n"
            "I should use natural Indonesian slang but not too much.\n"
            "Semoga harimu menyenangkan ya!"
        )
        cleaned = main.clean_answer(raw)
        self.assertNotIn("I should use natural Indonesian slang", cleaned)
        self.assertIn("Halo Kim Minji!", cleaned)
        self.assertIn("Semoga harimu menyenangkan ya!", cleaned)

    # -----------------------------------------------------------------------
    # Test B — Structured JSON output extraction
    # -----------------------------------------------------------------------
    def test_B_structured_json_reply_extracted(self) -> None:
        """Test B: JSON structured output — hanya nilai 'reply' yang dikembalikan."""
        raw = '{"reply": "Belum nih, Haimiya lagi sibuk nemenin kalian wkwk. Kamu sendiri udah makan belum?"}'
        result = main.extract_reply_from_json(raw)
        self.assertEqual(
            result,
            "Belum nih, Haimiya lagi sibuk nemenin kalian wkwk. Kamu sendiri udah makan belum?",
        )

    def test_B_json_in_code_fence_extracted(self) -> None:
        """Test B: JSON yang dibungkus ```json ... ``` tetap bisa diekstrak."""
        raw = '```json\n{"reply": "Halo! Ada yang bisa Haimiya bantu?"}\n```'
        result = main.extract_reply_from_json(raw)
        self.assertEqual(result, "Halo! Ada yang bisa Haimiya bantu?")

    def test_B_clean_answer_uses_json_first(self) -> None:
        """Test B: clean_answer memprioritaskan JSON parsing sebelum regex cleaning."""
        json_response = '{"reply": "Santai aja, Haimiya selalu di sini buat temenin kamu~"}'
        cleaned = main.clean_answer(json_response)
        self.assertEqual(cleaned, "Santai aja, Haimiya selalu di sini buat temenin kamu~")

    def test_B_full_json_not_sent_to_discord(self) -> None:
        """Test B: JSON object lengkap tidak boleh dikirim ke Discord; hanya value 'reply'."""
        json_response = '{"reply": "Hai! Senang ngobrol sama kamu."}'
        cleaned = main.clean_answer(json_response)
        self.assertNotIn("{", cleaned)
        self.assertNotIn('"reply"', cleaned)
        self.assertEqual(cleaned, "Hai! Senang ngobrol sama kamu.")

    # -----------------------------------------------------------------------
    # Test C — Separate thinking field never sent
    # -----------------------------------------------------------------------
    def test_C_separate_thinking_field_excluded(self) -> None:
        """Test C: message.thinking dari Ollama tidak pernah masuk ke jawaban atau history."""
        async def run():
            mock_response = AsyncMock()
            mock_response.message.thinking = (
                "Internal thought: user wants to know if I ate. "
                "I should respond naturally in Indonesian."
            )
            mock_response.message.content = '{"reply": "Belum nih wkwk, kamu udah makan?"}'
            mock_response.prompt_eval_count = 100
            mock_response.eval_count = 20

            with patch.object(main.ollama, "chat", AsyncMock(return_value=mock_response)):
                answer = await main.ask_ai(
                    channel_id=701,
                    guild_id=702,
                    user_id=703,
                    user_name="Kim Minji",
                    text="kamu udah makan?",
                    db_path=self.test_db_path,
                )

            self.assertNotIn("Internal thought", answer)
            self.assertNotIn("user wants to know", answer)
            self.assertEqual(answer, "Belum nih wkwk, kamu udah makan?")

            history = database.load_channel_history(701, 10, db_path=self.test_db_path)
            assistant_entries = [h for h in history if h["role"] == "assistant"]
            self.assertTrue(any("Belum nih wkwk" in h["content"] for h in assistant_entries))
            for h in assistant_entries:
                self.assertNotIn("Internal thought", h["content"])

        asyncio.run(run())

    # -----------------------------------------------------------------------
    # Test D — Database contamination prevention
    # -----------------------------------------------------------------------
    def test_D_contaminated_history_excluded_from_prompt(self) -> None:
        """Test D: Pesan history asisten yang terkontaminasi tidak boleh dimasukkan ke prompt."""
        with database.get_db_connection(self.test_db_path) as conn:
            conn.executemany(
                "INSERT INTO conversation_history (channel_id, user_id, display_name, role, content) VALUES (?, ?, ?, ?, ?)",
                [
                    (801, 1, "Tester", "user", "Halo Haimiya!"),
                    (801, 1, "Haimiya Senpai", "assistant",
                     "Okay, let's see. The user said halo. I should respond warmly.\n\nHai Tester!"),
                    (801, 1, "Tester", "user", "Siapa kamu?"),
                    (801, 1, "Haimiya Senpai", "assistant",
                     "We are in a conversation where Tester asked who I am.\n\nAku Haimiya Senpai!"),
                ],
            )

        history = database.load_channel_history(801, 20, db_path=self.test_db_path)

        for item in history:
            if item["role"] == "assistant":
                self.assertNotIn("Okay, let's see", item["content"],
                                 "Contaminated assistant message leaked into history")
                self.assertNotIn("We are in a conversation", item["content"],
                                 "Contaminated assistant message leaked into history")

    def test_D_clean_history_preserved(self) -> None:
        """Test D: Pesan history yang bersih tetap dipertahankan setelah sanitasi."""
        with database.get_db_connection(self.test_db_path) as conn:
            conn.executemany(
                "INSERT INTO conversation_history (channel_id, user_id, display_name, role, content) VALUES (?, ?, ?, ?, ?)",
                [
                    (802, 1, "Tester", "user", "Hai!"),
                    (802, 1, "Haimiya Senpai", "assistant", "Haii jugaa~ Ada yang mau diobrolkan?"),
                    (802, 1, "Tester", "user", "Rekomendasi anime dong."),
                    (802, 1, "Haimiya Senpai", "assistant", "Coba Horimiya, bagus banget!"),
                ],
            )

        history = database.load_channel_history(802, 20, db_path=self.test_db_path)
        self.assertEqual(len(history), 4)
        self.assertEqual(history[1]["content"], "Haii jugaa~ Ada yang mau diobrolkan?")
        self.assertEqual(history[3]["content"], "Coba Horimiya, bagus banget!")

    # -----------------------------------------------------------------------
    # Test E — Normal answers preserved
    # -----------------------------------------------------------------------
    def test_E_normal_conversation_answers_intact(self) -> None:
        """Test E: Jawaban percakapan normal tidak diubah oleh cleaner."""
        normal_cases = [
            "Hai! Senang ngobrol sama kamu~",
            "Belum nih, kamu sendiri udah makan?",
            "Lagi santai aja di server ini hehe.",
            "Coba tonton Horimiya, bagus banget ceritanya!",
            "GPU bekerja dengan memproses ribuan instruksi paralel melalui ribuan core kecil.",
            "Fungsi `async def` di Python digunakan untuk mendefinisikan coroutine yang bisa di-await.",
        ]
        for text in normal_cases:
            cleaned = main.clean_answer(text)
            self.assertEqual(cleaned, text, f"Normal answer should be preserved: {text!r}")

    # -----------------------------------------------------------------------
    # Test F — Invalid structured output / JSON parse failure
    # -----------------------------------------------------------------------
    def test_F_invalid_json_with_reasoning_uses_fallback(self) -> None:
        """Test F: JSON rusak + konten reasoning bocor → fallback bersih, bukan raw output."""
        async def run():
            mock_response = AsyncMock()
            mock_response.message.content = (
                "{ invalid json: Okay let's see. The user asked something. "
                "I need to respond as Haimiya. First I need to think about it."
            )
            mock_response.message.thinking = None
            mock_response.prompt_eval_count = 50
            mock_response.eval_count = 30

            with patch.object(main.ollama, "chat", AsyncMock(return_value=mock_response)):
                answer = await main.ask_ai(
                    channel_id=901,
                    guild_id=902,
                    user_id=903,
                    user_name="Tester",
                    text="Halo?",
                    db_path=self.test_db_path,
                )

            self.assertNotIn("Okay let's see", answer)
            self.assertNotIn("I need to respond", answer)
            self.assertNotIn("First I need to think", answer)
            self.assertTrue(len(answer) > 0)

        asyncio.run(run())

    def test_F_extract_reply_from_json_invalid_returns_none(self) -> None:
        """Test F: extract_reply_from_json mengembalikan None untuk input tidak valid."""
        invalid_cases = [
            "",
            "{}",
            '{"reply": ""}',
            "ini bukan JSON sama sekali",
            '{"other_field": "value"}',
            None,
        ]
        for case in invalid_cases:
            result = main.extract_reply_from_json(case)
            self.assertIsNone(result, f"Should return None for: {case!r}")

    # -----------------------------------------------------------------------
    # Test: ask_ai passes format=JSON schema to ollama.chat
    # -----------------------------------------------------------------------
    def test_ask_ai_passes_format_schema_to_ollama(self) -> None:
        """Pastikan ask_ai meneruskan tools di putaran 1 dan JSON schema format di putaran 2."""
        async def run():
            mock_tool_call = MagicMock()
            mock_tool_call.function.name = "web_search"
            mock_tool_call.function.arguments = {"query": "test"}

            mock_round1 = AsyncMock()
            mock_round1.message.tool_calls = [mock_tool_call]
            mock_round1.message.content = ""

            mock_round2 = AsyncMock()
            mock_round2.message.tool_calls = None
            mock_round2.message.content = '{"reply": "Halo!"}'

            mock_chat = AsyncMock(side_effect=[mock_round1, mock_round2])
            with patch.object(main.ollama, "chat", mock_chat):
                with patch.dict(main.TOOL_MAP, {"web_search": lambda query: "hasil pencarian"}):
                    await main.ask_ai(
                        channel_id=1001,
                        guild_id=1002,
                        user_id=1003,
                        user_name="Tester",
                        text="Halo?",
                        db_path=self.test_db_path,
                    )

            # Round 1 harus memuat parameter tools
            call_kwargs_round1 = mock_chat.call_args_list[0].kwargs
            self.assertIn("tools", call_kwargs_round1)

            # Round 2 harus memuat parameter format JSON schema
            call_kwargs_round2 = mock_chat.call_args_list[1].kwargs
            self.assertIn("format", call_kwargs_round2)
            fmt = call_kwargs_round2["format"]
            self.assertIsInstance(fmt, dict)
            self.assertEqual(fmt.get("type"), "object")
            self.assertIn("reply", fmt.get("properties", {}))

        asyncio.run(run())


if __name__ == "__main__":
    unittest.main()
