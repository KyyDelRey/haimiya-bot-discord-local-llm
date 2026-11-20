"""Unit and Integration Tests for Haimiya Senpai Modular Tools & 2-Step Tool Calling Loop.

Covers:
1. YouTube video ID extraction across formats
2. web_search output formatting & error handling
3. get_youtube_transcript output formatting, length truncation (5000 chars), & error handling
4. read_webpage HTML cleaning, length truncation (5000 chars), & error handling
5. 2-Step Tool Calling loop execution in ask_ai
6. Vision + Tool Calling combination
7. Status callback UX notifications
"""

import sys
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

# Tambahkan root directory ke sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# Sandbox compatibility patch for Windows
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
import tempfile
import unittest

import database
import main
import tools


class TestToolsModule(unittest.TestCase):
    """Test suite for functions in tools.py."""

    def test_extract_youtube_video_id(self) -> None:
        """Pastikan fungsi ekstraksi video ID bekerja di berbagai format URL YouTube."""
        cases = [
            ("https://www.youtube.com/watch?v=dQw4w9WgXcQ", "dQw4w9WgXcQ"),
            ("https://www.youtube.com/watch?v=dQw4w9WgXcQ&t=42s", "dQw4w9WgXcQ"),
            ("https://youtu.be/dQw4w9WgXcQ", "dQw4w9WgXcQ"),
            ("https://youtu.be/dQw4w9WgXcQ?t=10", "dQw4w9WgXcQ"),
            ("https://www.youtube.com/embed/dQw4w9WgXcQ", "dQw4w9WgXcQ"),
            ("https://www.youtube.com/shorts/dQw4w9WgXcQ", "dQw4w9WgXcQ"),
            ("dQw4w9WgXcQ", "dQw4w9WgXcQ"),
            ("https://example.com/not-youtube", None),
            ("invalid", None),
        ]
        for url, expected in cases:
            self.assertEqual(
                tools.extract_youtube_video_id(url),
                expected,
                f"Gagal mengekstrak ID dari '{url}'",
            )

    @patch("ddgs.DDGS")
    def test_web_search_success(self, mock_ddgs_cls) -> None:
        """Uji web_search saat DuckDuckGo mengembalikan hasil pencarian."""
        mock_instance = MagicMock()
        mock_instance.__enter__.return_value = mock_instance
        mock_instance.text.return_value = [
            {
                "title": "Hasil 1",
                "body": "Ringkasan informasi pertama.",
                "href": "https://example.com/1",
            },
            {
                "title": "Hasil 2",
                "body": "Ringkasan informasi kedua.",
                "href": "https://example.com/2",
            },
        ]
        mock_ddgs_cls.return_value = mock_instance

        res = tools.web_search("test query")
        self.assertIn("Hasil 1", res)
        self.assertIn("Ringkasan informasi pertama.", res)
        self.assertIn("https://example.com/1", res)
        self.assertIn("Hasil 2", res)

    @patch("ddgs.DDGS")
    def test_web_search_empty_and_error(self, mock_ddgs_cls) -> None:
        """Uji web_search saat hasil kosong atau terjadi exception."""
        mock_instance = MagicMock()
        mock_instance.__enter__.return_value = mock_instance
        mock_instance.text.return_value = []
        mock_ddgs_cls.return_value = mock_instance

        empty_res = tools.web_search("query kosong")
        self.assertIn("Tidak ditemukan", empty_res)

        mock_instance.text.side_effect = RuntimeError("DDG Blocked")
        error_res = tools.web_search("error query")
        self.assertIn("Gagal melakukan pencarian", error_res)

    @patch("youtube_transcript_api.YouTubeTranscriptApi")
    def test_get_youtube_transcript_success_and_truncate(self, mock_ytt_cls) -> None:
        """Uji get_youtube_transcript berhasil dan memotong teks di batas 5000 karakter."""
        mock_snippet_1 = MagicMock()
        mock_snippet_1.text = "Halo semuanya, selamat datang di video ini."
        # Buat snippet panjang untuk menguji batasan 5000 karakter
        mock_snippet_2 = MagicMock()
        mock_snippet_2.text = "Konten sangat panjang. " * 350

        mock_instance = MagicMock()
        mock_instance.fetch.return_value = [mock_snippet_1, mock_snippet_2]
        mock_ytt_cls.return_value = mock_instance

        res = tools.get_youtube_transcript("https://youtu.be/dQw4w9WgXcQ")
        self.assertIn("Transkrip Video YouTube (dQw4w9WgXcQ):", res)
        self.assertIn("Halo semuanya", res)
        self.assertIn("dipotong karena batas maksimum 5000 karakter", res)
        self.assertLessEqual(len(res), 5200)

    def test_get_youtube_transcript_invalid_url(self) -> None:
        """Uji URL tidak valid ditangani tanpa crash."""
        res = tools.get_youtube_transcript("bukan-youtube")
        self.assertIn("Error: URL YouTube tidak valid", res)

    @patch("requests.get")
    def test_read_webpage_success_and_html_cleaning(self, mock_requests_get) -> None:
        """Uji read_webpage membersihkan script/style/nav dan mengekstrak konten utama."""
        html_content = """
        <html>
            <head><title>Artikel Keren</title><style>body { color: red; }</style></head>
            <body>
                <header><nav><a href="/">Menu</a></nav></header>
                <article>
                    <h1>Berita Terkini</h1>
                    <p>Ini adalah isi artikel penting.</p>
                </article>
                <script>console.log("ads");</script>
                <footer>Hak Cipta 2026</footer>
            </body>
        </html>
        """
        mock_resp = MagicMock()
        mock_resp.text = html_content
        mock_resp.status_code = 200
        mock_requests_get.return_value = mock_resp

        res = tools.read_webpage("https://example.com/berita")
        self.assertIn("Judul Web: Artikel Keren", res)
        self.assertIn("Ini adalah isi artikel penting.", res)
        self.assertNotIn("console.log", res)
        self.assertNotIn("body { color: red; }", res)
        self.assertNotIn("Hak Cipta 2026", res)


class TestToolCallingLoop(unittest.TestCase):
    """Integration test suite for the 2-step tool calling loop in ask_ai."""

    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.test_db_path = Path(self.temp_dir.name) / "test_tool_calling.db"
        database.init_database(self.test_db_path)

    def tearDown(self) -> None:
        self.temp_dir.cleanup()

    def test_two_step_tool_calling_loop_execution(self) -> None:
        """Uji alur 2 putaran: Round 1 memanggil tool_calls -> eksekusi tools -> Round 2 menghasilkan jawaban akhir."""
        async def run_test():
            # Mock Round 1: Model memutuskan untuk memanggil tool web_search
            mock_tool_call = MagicMock()
            mock_tool_call.function.name = "web_search"
            mock_tool_call.function.arguments = {"query": "jadwal rilis anime 2026"}

            mock_round1_resp = MagicMock()
            mock_round1_resp.message.tool_calls = [mock_tool_call]
            mock_round1_resp.message.content = ""
            mock_round1_resp.prompt_eval_count = 100
            mock_round1_resp.eval_count = 20

            # Mock Round 2: Model menghasilkan jawaban akhir setelah melihat hasil tool
            mock_round2_resp = MagicMock()
            mock_round2_resp.message.tool_calls = None
            mock_round2_resp.message.content = '{"reply": "Berdasarkan pencarian, beberapa anime baru akan rilis di awal tahun 2026 nih!"}'
            mock_round2_resp.prompt_eval_count = 150
            mock_round2_resp.eval_count = 30

            status_updates = []

            async def mock_status_callback(status: str):
                status_updates.append(status)

            # Patch ollama.chat untuk mengembalikan mock_round1 lalu mock_round2
            mock_chat = AsyncMock(side_effect=[mock_round1_resp, mock_round2_resp])

            with patch.object(main.ollama, "chat", mock_chat):
                with patch.object(tools, "web_search", return_value="1. Anime Baru 2026 - Rilis Januari 2026"):
                    # Update mapping di main agar mock terpanggil
                    with patch.dict(main.TOOL_MAP, {"web_search": lambda query: "1. Anime Baru 2026 - Rilis Januari 2026"}):
                        answer = await main.ask_ai(
                            channel_id=301,
                            guild_id=401,
                            user_id=501,
                            user_name="AnimeFan",
                            text="cariin jadwal anime rilis 2026 dong",
                            status_callback=mock_status_callback,
                            db_path=self.test_db_path,
                        )

            # Verifikasi jawaban akhir
            self.assertIn("anime baru akan rilis di awal tahun 2026", answer)

            # Verifikasi bahwa ollama.chat dipanggil sebanyak 2 kali (2-step loop)
            self.assertEqual(mock_chat.call_count, 2)

            # Verifikasi Round 1 memuat tools
            first_call_kwargs = mock_chat.call_args_list[0].kwargs
            self.assertIn("tools", first_call_kwargs)
            self.assertEqual(first_call_kwargs["tools"], tools.AVAILABLE_TOOLS)

            # Verifikasi Round 2 menerima messages yang memuat pesan tool
            second_call_kwargs = mock_chat.call_args_list[1].kwargs
            sent_messages = second_call_kwargs["messages"]
            tool_role_messages = [m for m in sent_messages if isinstance(m, dict) and m.get("role") == "tool"]
            self.assertTrue(len(tool_role_messages) >= 1)
            self.assertEqual(tool_role_messages[0]["tool_name"], "web_search")
            self.assertIn("Anime Baru 2026", tool_role_messages[0]["content"])

            # Verifikasi status_callback menerima notifikasi UX
            self.assertTrue(len(status_updates) >= 2)
            self.assertTrue(any("Mencari informasi" in s for s in status_updates))

        asyncio.run(run_test())

    def test_vision_combined_with_tool_calling(self) -> None:
        """Uji gambar (vision) dipertahankan saat tool calling dijalankan."""
        async def run_test():
            fake_image_bytes = b"\x89PNG\r\n\x1a\nfakeimagebytes"

            mock_tool_call = MagicMock()
            mock_tool_call.function.name = "web_search"
            mock_tool_call.function.arguments = {"query": "identifikasi objek anime"}

            mock_round1_resp = MagicMock()
            mock_round1_resp.message.tool_calls = [mock_tool_call]
            mock_round1_resp.message.content = ""

            mock_round2_resp = MagicMock()
            mock_round2_resp.message.tool_calls = None
            mock_round2_resp.message.content = '{"reply": "Dari gambar dan hasil pencarian, ini adalah karakter Haimiya Senpai!"}'

            mock_chat = AsyncMock(side_effect=[mock_round1_resp, mock_round2_resp])

            with patch.object(main.ollama, "chat", mock_chat):
                with patch.dict(main.TOOL_MAP, {"web_search": lambda query: "Info: Karakter Haimiya"}):
                    answer = await main.ask_ai(
                        channel_id=302,
                        guild_id=402,
                        user_id=502,
                        user_name="Otaku",
                        text="Ini siapa ya? Cariin infonya",
                        image_bytes_list=[fake_image_bytes],
                        db_path=self.test_db_path,
                    )

            self.assertIn("karakter Haimiya Senpai", answer)

            # Pastikan pesan user di Round 1 memuat field images
            first_call_messages = mock_chat.call_args_list[0].kwargs["messages"]
            user_msg = [m for m in first_call_messages if isinstance(m, dict) and m.get("role") == "user"][-1]
            self.assertIn("images", user_msg)
            self.assertEqual(user_msg["images"], [fake_image_bytes])

        asyncio.run(run_test())


if __name__ == "__main__":
    unittest.main()
