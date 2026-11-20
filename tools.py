"""Modular tools for Haimiya Senpai Discord Bot.

Provides tools for native function calling with Ollama:
- web_search: Real-time internet search via DuckDuckGo (DDGS)
- get_youtube_transcript: YouTube subtitles/transcript extraction
- read_webpage: Webpage article text extraction via requests & BeautifulSoup4
"""

import re
from typing import Any


def extract_youtube_video_id(url: str) -> str | None:
    """Ekstrak 11-karakter video ID dari berbagai format URL YouTube."""
    stripped = url.strip()
    # Jika langsung berupa 11 karakter ID murni tanpa awalan URL
    if len(stripped) == 11 and re.match(r"^[0-9A-Za-z_-]{11}$", stripped) and not stripped.startswith("http"):
        return stripped

    patterns = [
        r"(?:https?:\/\/)?(?:www\.|m\.)?youtube\.com\/watch\?.*v=([0-9A-Za-z_-]{11})",
        r"(?:https?:\/\/)?youtu\.be\/([0-9A-Za-z_-]{11})",
        r"(?:https?:\/\/)?(?:www\.|m\.)?youtube\.com\/embed\/([0-9A-Za-z_-]{11})",
        r"(?:https?:\/\/)?(?:www\.|m\.)?youtube\.com\/shorts\/([0-9A-Za-z_-]{11})",
        r"(?:https?:\/\/)?(?:www\.|m\.)?youtube\.com\/v\/([0-9A-Za-z_-]{11})",
    ]
    for pattern in patterns:
        match = re.search(pattern, stripped)
        if match:
            return match.group(1)
    return None


def web_search(query: str) -> str:
    """Cari informasi terbaru dan real-time di internet menggunakan DuckDuckGo.

    Args:
        query: Kata kunci pencarian atau pertanyaan untuk mencari informasi terbaru di web.

    Returns:
        str: Hasil pencarian teratas (maksimal 4-5) berisi judul, ringkasan, dan URL sumber.
    """
    try:
        try:
            from ddgs import DDGS
        except ImportError:
            from duckduckgo_search import DDGS

        with DDGS() as ddgs:
            results = list(ddgs.text(query, max_results=5))

        if not results:
            return f"Tidak ditemukan hasil pencarian untuk: '{query}'"

        formatted = []
        for i, r in enumerate(results[:5], 1):
            title = r.get("title", "Tanpa Judul")
            snippet = r.get("body", "")
            url = r.get("href", "")
            formatted.append(f"{i}. {title}\n   Ringkasan: {snippet}\n   Sumber: {url}")

        return "\n\n".join(formatted)
    except Exception as e:
        return f"Gagal melakukan pencarian web untuk '{query}': {e}"


def get_youtube_transcript(url: str) -> str:
    """Ambil transkrip atau subtitle dari video YouTube berdasarkan URL atau Video ID.

    Args:
        url: URL video YouTube lengkap (misal: https://www.youtube.com/watch?v=... atau https://youtu.be/...) atau 11-karakter video ID.

    Returns:
        str: Teks transkrip video YouTube (maksimal 5000 karakter).
    """
    video_id = extract_youtube_video_id(url)
    if not video_id:
        return f"Error: URL YouTube tidak valid atau Video ID tidak ditemukan dari '{url}'."

    try:
        from youtube_transcript_api import YouTubeTranscriptApi

        ytt = YouTubeTranscriptApi()

        # Prioritaskan bahasa Indonesia ('id'), lalu bahasa Inggris ('en')
        try:
            fetched = ytt.fetch(video_id, languages=("id", "en"))
        except Exception:
            # Fallback jika 'id' dan 'en' tidak ada secara eksplisit: coba daftar transkrip yang ada
            transcript_list = ytt.list(video_id)
            try:
                t = transcript_list.find_transcript(["id", "en"])
            except Exception:
                # Ambil transkrip pertama yang tersedia
                t = next(iter(transcript_list))
            fetched = t.fetch()

        lines = [snippet.text.strip() for snippet in fetched if snippet.text.strip()]
        full_text = " ".join(lines)
        if not full_text:
            return f"Transkrip video YouTube ({video_id}) kosong atau tidak ditemukan."

        if len(full_text) > 5000:
            full_text = full_text[:5000] + "\n\n... [Transkrip dipotong karena batas maksimum 5000 karakter]"

        return f"Transkrip Video YouTube ({video_id}):\n{full_text}"
    except Exception as e:
        return f"Gagal mengambil transkrip YouTube untuk video '{video_id}': {e}"


def read_webpage(url: str) -> str:
    """Baca dan ekstrak teks konten utama dari suatu URL halaman web atau artikel.

    Args:
        url: Alamat website lengkap (URL) yang ingin dibaca (misal: https://example.com/artikel).

    Returns:
        str: Teks konten bersih dari halaman web (maksimal 5000 karakter).
    """
    cleaned_url = url.strip()
    if not cleaned_url.startswith(("http://", "https://")):
        cleaned_url = "https://" + cleaned_url

    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/124.0.0.0 Safari/537.36"
        ),
        "Accept-Language": "id,en-US;q=0.9,en;q=0.8",
    }

    try:
        import requests
        from bs4 import BeautifulSoup

        response = requests.get(cleaned_url, headers=headers, timeout=12)
        response.raise_for_status()

        soup = BeautifulSoup(response.text, "html.parser")

        # Bersihkan elemen yang tidak relevan
        for tag in soup(["script", "style", "nav", "header", "footer", "aside", "form", "noscript", "svg", "iframe"]):
            tag.decompose()

        # Prioritaskan elemen konten utama jika tersedia
        main_content = soup.find("article") or soup.find("main") or soup.body or soup

        raw_text = main_content.get_text(separator="\n", strip=True)
        cleaned_lines = [line.strip() for line in raw_text.splitlines() if line.strip()]
        text = "\n".join(cleaned_lines)

        if not text:
            return f"Halaman web '{cleaned_url}' tidak memiliki konten teks yang dapat dibaca."

        if len(text) > 5000:
            text = text[:5000] + "\n\n... [Konten dipotong karena batas maksimum 5000 karakter]"

        title = soup.title.string.strip() if soup.title and soup.title.string else "Tanpa Judul"
        return f"Judul Web: {title}\nURL: {cleaned_url}\n\nKonten:\n{text}"
    except requests.exceptions.Timeout:
        return f"Gagal membaca web '{cleaned_url}': Waktu koneksi habis (timeout)."
    except requests.exceptions.HTTPError as e:
        status_code = e.response.status_code if e.response else "Unknown"
        return f"Gagal membaca web '{cleaned_url}': HTTP Error {status_code}."
    except Exception as e:
        return f"Gagal membaca web '{cleaned_url}': {e}"


# List tools yang akan diteruskan ke Ollama
AVAILABLE_TOOLS = [web_search, get_youtube_transcript, read_webpage]

# Mapping nama fungsi ke callable
TOOL_MAP: dict[str, Any] = {tool.__name__: tool for tool in AVAILABLE_TOOLS}
