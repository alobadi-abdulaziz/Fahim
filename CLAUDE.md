# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Running the App

```bash
python app.py
# Opens at http://localhost:5000
```

Requires:
- Python venv activated with `pip install -r requirements.txt`
- ffmpeg at `C:\Users\oscar\Desktop\ffmpeg-8.1-essentials_build\bin`
- `sshi_metadata_try1.json` and `embeddings_index.npz` in root
- Sign language videos in `data/videos/`

Startup takes ~30s (loads Whisper model + sentence-transformers + embeddings index).

## Architecture

**Arabic text → sign language video pipeline:**

```
User input (text or voice)
  → Phrase-first match (n-gram exact lookup against dictionary)
  → Remaining words: DeepSeek converts dialect → Fusha synonyms
  → Per-word: exact match → embedding search → DeepSeek judge picks best
  → Proper names: fingerspelled letter-by-letter
  → Collect video IDs → ffmpeg concat → merged video plays in chat UI
```

### Key Files

- `app.py` — Flask server. Routes: `/chat` (text→video), `/voice` (mic→text via Whisper), `/video` (serve merged mp4). Handles video merging with ffmpeg `-c copy`.
- `search.py` — Core search engine. Multi-layer: phrase matching → exact search (with Arabic normalization) → embedding similarity → DeepSeek LLM judge. Manages all data lookups and the DeepSeek API.
- `index.html` — RTL Arabic chat UI (dark theme). Vanilla JS, no framework.
- `sshi_metadata_try1.json` — Dictionary: 7,841 entries (5,680 single-word, 2,161 phrases). Fields: `id`, `wordAr`, `synonym`, `description`, `video_filename`.
- `embeddings_index.npz` — Pre-computed embeddings for all `wordAr` values. Delete to regenerate.

### Search Pipeline Details (search.py)

1. **Phrase-first** (`phrase_match_spans`): slides n-grams (5→2 words) over sentence, exact-matches against `word_to_id`. Locked spans skip the per-word pipeline.
2. **DeepSeek synonyms**: converts each remaining word to up to 7 Fusha equivalents. Names are detected and returned as individual letters for fingerspelling.
3. **Per-word search** (`search_word`): tries exact match on all fusha candidates → embedding cosine similarity (top-k) → DeepSeek judge selects best from all collected candidates. Judge sees `wordAr`, `synonym`, and `description` per candidate.
4. **Normalization** (`normalize`): strips ال prefix, unifies hamzas (أإآ→ا), merges ة→ه and ى→ي, removes tashkeel and tatweel.

### External Dependencies

- **DeepSeek API** (`api.deepseek.com`): used for synonym extraction and final candidate judging. Two API calls per unmatched word.
- **Whisper** (local, `small` model): Arabic speech-to-text for voice input.
- **Sentence-transformers** (`paraphrase-multilingual-MiniLM-L12-v2`): embedding model for semantic search fallback.
- **ffmpeg**: video concatenation. All videos must share format (1920x1080, 25fps, H.264 High) for `-c copy` to work without lag.

### Video Format Constraint

All videos in `data/videos/` must be 1920x1080, 25fps, H.264 High profile. Letter videos (IDs 1-38) were batch-converted to match word videos. If new videos with different formats are added, they must be converted first or the concat will produce playback glitches.
