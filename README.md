# Arabic Sign Language Translator

A web app that translates Arabic text or speech into sign language videos. Type a sentence (or speak it), and the app finds the matching sign for each word and plays them as one continuous video.

## Features

- **Text input** — type Arabic sentences (dialect or Fusha)
- **Voice input** — speak into the mic, Whisper converts to text
- **Phrase matching** — recognizes multi-word expressions (e.g. "السلام عليكم") as single signs
- **Name fingerspelling** — proper names are spelled letter-by-letter using sign alphabet
- **Dialect support** — converts dialectal Arabic to Fusha before searching the dictionary

## How It Works

```
Arabic sentence
  → Phrase-first match (multi-word dictionary lookup)
  → DeepSeek AI converts remaining words to Fusha synonyms
  → Search: exact match → semantic embeddings → AI judge picks best
  → Proper names → fingerspelled letter by letter
  → ffmpeg merges sign videos into one clip
  → Plays in chat UI
```

## Setup

### Prerequisites

- Python 3.10+
- [ffmpeg](https://ffmpeg.org/download.html) — download and extract to `C:\Users\oscar\Desktop\ffmpeg-8.1-essentials_build\`
- DeepSeek API key (configured in `search.py`)

### Install

```bash
python -m venv .venv
source .venv/Scripts/activate   # Windows (Git Bash)
pip install -r requirements.txt
```

### Run

```bash
python app.py
```

Open http://localhost:5000

## Project Structure

```
app.py                  — Flask server (routes, video merge, Whisper)
search.py               — Search engine (phrase match, embeddings, DeepSeek AI)
index.html              — Chat UI (RTL Arabic, dark theme)
sshi_metadata_try1.json — Dictionary (7,841 sign entries with synonyms & descriptions)
embeddings_index.npz    — Pre-computed word embeddings cache
data/videos/            — Sign language video clips (.mp4)
requirements.txt        — Python dependencies
```

## Dictionary

The sign language dictionary (`sshi_metadata_try1.json`) contains:
- 5,680 single-word entries
- 2,161 multi-word phrase entries
- 3,379 entries with synonyms
- 7,323 entries with descriptions

Each entry maps an Arabic word/phrase to a video file showing the corresponding sign.

## Tech Stack

- **Flask** — web server
- **OpenAI Whisper** — Arabic speech-to-text
- **Sentence-Transformers** — semantic search (paraphrase-multilingual-MiniLM-L12-v2)
- **DeepSeek API** — dialect→Fusha conversion and candidate selection
- **ffmpeg** — video concatenation
