# 🌸 Haimiya Senpai — Discord AI Companion Bot (Phase 2)

Haimiya Senpai adalah AI companion Discord bertema anime yang ramah, hangat, playful, dan sedikit teasing. Bot ini terintegrasi langsung dengan LLM lokal (**Qwen3.5:4B**) via **Ollama**, dilengkapi dengan **Mood System**, **Relationship System**, **Personal Memory**, dan **Channel Conversation History** berbasis **SQLite WAL**.

---

## 🚀 Fitur Utama

### 1. Dynamic Mood System (Phase 2)

Haimiya memiliki kondisi emosional dinamis yang memengaruhi gaya bicara tanpa mengubah kepribadian intinya:

- **Happy (✨):** Ceria, ramah, energik, dan antusias menyambut obrolan.
- **Playful (😜):** Suka bercanda, cerdik, dan sedikit teasing secara bersahabat.
- **Curious (🔍):** Banyak bertanya dan tertarik mendalami topik pembicaraan.
- **Sleepy (💤):** Mengantuk, sedikit malas, sesekali menguap (_hoam..._), respon santai dan ringkas.
- **Calm (🍵):** Tenang, santai, adem, damai, dan bijak.
- **Shy (🌸):** Sedikit pemalu dan tersipu (_u-umm..._), tetapi tetap ramah dan manis.

> **Karakteristik Mood:**
>
> - Bertahan selama rentang durasi tertentu (default `30`–`60` menit) sebelum berganti.
> - **Tidak berganti setiap pesan.**
> - Transisi alami berdasarkan matriks transisi emosi dan kata kunci konteks percakapan serta waktu lokal.
> - Tetap tersimpan di database (`guild_moods`) sehingga status mood tidak hilang saat bot restart.
> - Bot bersifat **reaktif** dan tidak pernah mengirim pesan acak sendiri saat mood berganti.

### 2. Relationship System (Phase 2)

Setiap member memiliki tingkat kedekatan personal tersendiri dengan Haimiya (scope per `guild_id` & `user_id`):
| Level | Affection Points (AP) | Status | Sikap Haimiya |
|---|---|---|---|
| **1** | 0–19 AP | **Acquaintance** (🌱) | Ramah, sopan, menjaga batas wajar baru berkenalan. |
| **2** | 20–59 AP | **Familiar** (🌿) | Lebih santai, mulai akrab dan mengenali user. |
| **3** | 60–119 AP | **Close Friend** (⭐) | Teman dekat, sering bercanda dan menimpali obrolan seru. |
| **4** | 120–199 AP | **Bestie** (💖) | Sahabat karib, sangat nyaman, playful, dan suka teasing. |
| **5** | 200+ AP | **Special** (👑) | Hubungan istimewa, sangat hangat, penuh rasa saling percaya. |

> **Aturan Affection Points:**
>
> - Poin diperoleh **hanya ketika interaksi AI berhasil**.
> - **Cooldown:** 60 detik (default) untuk mencegah spamming.
> - **Daily Cap:** Maksimal 20 AP per hari (default). Otomatis ter-reset setiap hari baru.
> - Pesan bot, pesan error, dan pesan kosong tidak mendapatkan poin.

### 3. Personal Memory System

- Mengingat preferensi, hobi, dan catatan personal member secara rahasia per guild.
- Digunakan secara halus untuk mempersonalisasi jawaban tanpa membocorkannya ke member lain.

### 4. Natural Conversation, Vision & Native Tool Calling

- **Vision Multimodal:** Mampu membaca, mengenali, dan mengomentari gambar yang dikirimkan user via mention, reply, atau attachment `/ask`.
- **Native Tool Calling (Function Calling):**
  - `web_search`: Pencarian web informasi terkini dan real-time menggunakan DuckDuckGo.
  - `get_youtube_transcript`: Ekstraksi subtitle / transkrip video YouTube otomatis (prioritas bahasa `'id'` dan `'en'`).
  - `read_webpage`: Ekstraksi konten artikel dari halaman web / URL eksternal dengan pembersihan HTML otomatis.
  - Batasan context window otomatis (maksimal ~5000 karakter per tool) untuk menjaga kestabilan memori.
- **2-Step Asynchronous Loop:** Mengeksekusi tools via `asyncio.to_thread` tanpa membuat bot lag atau disconnect, dilengkapi indikator status interaktif di Discord (`🔍 Mencari informasi...`, dll).
- Menyimpan riwayat percakapan channel dengan limit pruning otomatis.
- Merespons **mention** (`@Haimiya Senpai`) dan **reply** pesan bot tanpa perlu mention ulang.
- Membersihkan tag pemikiran model (`<think>`) secara otomatis.

---

## 🛠️ Persyaratan Sistem

- **Python:** 3.10+ (Direkomendasikan Python 3.11 – 3.14)
- **Discord Library:** `discord.py >= 2.4`
- **Ollama:** Versi terbaru dengan model `qwen3:4b`
- **Database:** SQLite (Built-in Python dengan WAL mode)

---

## 📦 Instalasi

1. **Clone repository ini ke komputer kamu.**
2. **Install dependency yang dibutuhkan:**
   ```bash
   pip install -r Requirements.txt
   ```
3. **Pastikan Ollama sudah berjalan dan model `qwen3:4b` terunduh:**
   ```bash
   ollama pull qwen3:4b
   ollama serve
   ```
4. **Konfigurasi file `.env`:**
   Sesuaikan token bot Discord dan konfigurasi lainnya di file `.env`:

   ```env
   DISCORD_TOKEN=your_discord_bot_token_here
   OLLAMA_HOST=http://localhost:11434
   OLLAMA_MODEL=qwen3.5:4b
   OLLAMA_NUM_CTX=4096
   OLLAMA_MAX_TOKENS=600
   OLLAMA_THINK=false
   DATABASE_PATH=data/haimiya.db
   MAX_HISTORY=12

   # Phase 2: Mood System Configuration
   MOOD_MIN_DURATION_MINUTES=30
   MOOD_MAX_DURATION_MINUTES=60

   # Phase 2: Relationship System Configuration
   RELATIONSHIP_POINTS_PER_INTERACTION=1
   RELATIONSHIP_COOLDOWN_SECONDS=60
   RELATIONSHIP_DAILY_CAP=20
   ```

---

## 🎮 Menjalankan Bot

Jalankan bot menggunakan Python:

```bash
python main.py
```

Saat bot berjalan:

- Database dan tabel SQLite akan diinisialisasi otomatis tanpa menghapus data yang ada.
- Seluruh slash command akan disinkronkan ke Discord Developer Portal.
- Bot akan login dan menampilkan status online di terminal.

---

## 📜 Daftar Slash Commands

### Percakapan & AI

- `/ask pertanyaan:<teks>`: Mengobrol langsung dengan Haimiya Senpai.
- `/reset`: Menghapus riwayat percakapan channel saat ini (catatan personal member tetap aman).

### Mood & Relationship (Phase 2)

- `/mood`: Menampilkan kondisi mood Haimiya di server saat ini, efek pada gaya bicara, dan sisa durasi.
- `/relationship [rahasia:bool]`: Menampilkan kartu kedekatan kamu dengan Haimiya (Level, Status, Total AP, Progress Bar, Kuota Hari Ini).
- `/relationship_admin view member:@User`: _(Khusus Admin)_ Melihat relationship member lain.
- `/relationship_admin set_points member:@User points:<int>`: _(Khusus Admin)_ Menetapkan poin member secara manual.
- `/relationship_admin add_points member:@User delta:<int>`: _(Khusus Admin)_ Menambah atau mengurangi poin member.
- `/relationship_admin reset member:@User`: _(Khusus Admin)_ Mereset relationship member kembali ke Level 1 (0 AP).

### Personal Memory

- `/remember informasi:<teks>`: Menyimpan catatan personal kamu ke memori Haimiya.
- `/memory`: Melihat catatan personal yang diingat Haimiya tentang kamu.
- `/forget`: Menghapus seluruh catatan personal kamu dari memori Haimiya.

---

## 🧪 Menjalankan Automated Test Suite

Untuk memverifikasi fungsionalitas sistem (database, mood, level, cooldown, anti-reasoning, vision, serta modular tools dan 2-step tool calling loop):

```bash
python -m unittest discover -s tests -v
```

Seluruh 37 unit test dirancang mandiri dan aman dieksekusi di environment lokal tanpa memerlukan akses live bot Discord, mencakup pengujian database, persistensi, mood, relationship, anti-reasoning, multimodal vision, hingga web scraping, DuckDuckGo search, dan YouTube transcripts.
