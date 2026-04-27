# Arabic Sign Language Translator — سنج لن

A multi-page web platform that translates Arabic text or speech into sign language videos. Type a sentence (or speak it), and the app finds the matching sign for each word and plays them as one continuous video.

## Features

- **Text input** — type Arabic sentences (dialect or Fusha)
- **Voice input** — speak into the mic, Whisper converts to text
- **Phrase matching** — recognizes multi-word expressions (e.g. "السلام عليكم") as single signs
- **Name fingerspelling** — proper names are spelled letter-by-letter using sign alphabet
- **Dialect support** — converts dialectal Arabic to Fusha before searching the dictionary
- **Learning platform** — level-based sign language lessons (beginner to expert)
- **Live streaming view** — continuous speech-to-sign translation layout

## How It Works

```
Arabic sentence
  → Phrase-first match (multi-word dictionary lookup)
  → DeepSeek AI converts remaining words to Fusha synonyms
  → Parallel search: exact match ║ semantic embeddings (simultaneous threads)
  → DeepSeek AI judge picks best candidate given full sentence context
  → Proper names → fingerspelled letter by letter
  → ffmpeg merges sign videos into one clip
  → Plays in chat UI
```

## Setup

### Prerequisites

- Python 3.10+
- [ffmpeg](https://ffmpeg.org/download.html) — extract to `C:\Users\oscar\Desktop\ffmpeg-8.1-essentials_build\`
- DeepSeek API key (set in `search.py`)
- Sign language video files in `data/videos/` (not included in repo — too large)
- `sshi_metadata_try1.json` dictionary file in root (not included — too large)
- `embeddings_index.npz` embeddings cache in root (auto-generated on first run if missing)

### Install

```bash
python -m venv .venv
source .venv/Scripts/activate   # Windows Git Bash
# or: .venv\Scripts\activate    # Windows CMD
pip install -r requirements.txt
```

### Run

```bash
python app.py
```

Open http://localhost:5000 — startup takes ~30s (loads Whisper + sentence-transformers + embeddings).

## Project Structure

```
app.py                      — Flask server (routes, video merge, Whisper)
search.py                   — Search engine (phrase match, embeddings, DeepSeek AI)
templates/
  base.html                 — Shared Jinja2 base (navbar, footer)
  index.html                — Landing page
  translator.html           — Chat / translation UI
  learning.html             — Learning platform (level cards)
  streaming.html            — Live streaming layout
static/
  css/style.css             — Basira design system (gold-on-teal dark theme)
  js/translator.js          — Chat UI logic
sshi_metadata_try1.json     — Dictionary: 7,841 sign entries (excluded from repo)
embeddings_index.npz        — Pre-computed word embeddings cache (excluded from repo)
data/videos/                — Sign language video clips (excluded from repo)
requirements.txt            — Python dependencies
```

## Pages

| Route | Page |
|---|---|
| `/` | Landing page with hero and feature overview |
| `/translator` | Main chat translation interface |
| `/learning` | Level-based sign language courses |
| `/streaming` | Continuous live translation layout |

## Dictionary

The sign language dictionary (`sshi_metadata_try1.json`) contains:
- 5,680 single-word entries
- 2,161 multi-word phrase entries
- Each entry maps an Arabic word/phrase to a video file, with optional synonyms and description

## Tech Stack

| Component | Technology |
|---|---|
| Web server | Flask |
| Speech-to-text | OpenAI Whisper (`small` model, local) |
| Semantic search | Sentence-Transformers (`intfloat/multilingual-e5-base`) |
| AI translation | DeepSeek API (`deepseek-chat`) |
| Video merging | ffmpeg (`-c copy` stream concat) |
| Frontend | Vanilla HTML/CSS/JS, Jinja2 templates, RTL Arabic |
