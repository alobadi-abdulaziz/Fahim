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
  → Per-word: exact match ║ embedding search (parallel threads)
  → DeepSeek judge picks best candidate from combined results
  → Proper names: fingerspelled letter-by-letter
  → Collect video IDs → ffmpeg concat → merged video plays in chat UI
```

### Key Files

- `app.py` — Flask server. Routes: `/` (landing), `/translator`, `/learning`, `/streaming`, `/chat` (text→video), `/voice` (mic→text via Whisper), `/video` (serve merged mp4). Uses `render_template` for all pages.
- `search.py` — Core search engine. Multi-layer: phrase matching → parallel exact+embedding search → DeepSeek LLM judge. Manages all data lookups and the DeepSeek API.
- `templates/base.html` — Jinja2 base template. Shared navbar (3 pillars + logo), footer, CSS/JS includes.
- `templates/index.html` — Landing page. Hero section + 4-card feature grid.
- `templates/translator.html` — Chat UI. Extends base, references `static/js/translator.js`.
- `templates/learning.html` — Learning platform. Level cards with progress bars.
- `templates/streaming.html` — Live streaming stub. Split video+transcript layout with start/stop toggle.
- `static/css/style.css` — Full Basira design system. Deep teal `#06242b` + sand gold `#c2a378` palette, diagonal grid background, glass navbar, all shared components.
- `static/js/translator.js` — Chat UI logic (send, voice, video playback). Extracted from old inline index.html.
- `sshi_metadata_try1.json` — Dictionary: 7,841 entries (5,680 single-word, 2,161 phrases). Fields: `id`, `wordAr`, `synonym`, `description`, `video_filename`.
- `embeddings_index.npz` — Pre-computed embeddings for all `wordAr` values. Delete to regenerate.

### Search Pipeline Details (search.py)

1. **Phrase-first** (`phrase_match_spans`): slides n-grams (5→2 words) over sentence, exact-matches against `word_to_id`. Locked spans skip the per-word pipeline.
2. **DeepSeek synonyms**: converts each remaining word to up to 7 Fusha equivalents. Names are detected and returned as individual letters for fingerspelling.
3. **Per-word search** (`search_word`): runs Step A (exact match) and Step B (embedding cosine similarity, top-k) **in parallel threads**. If only one candidate found, returns immediately. Otherwise Step C: DeepSeek judge selects best from all candidates using full sentence context. Judge sees `wordAr`, `synonym`, and `description` per candidate.
4. **Normalization** (`normalize`): strips ال prefix, unifies hamzas (أإآ→ا), merges ة→ه and ى→ي, removes tashkeel and tatweel.

### Frontend Structure

Multi-page Flask app with Jinja2 templates. All pages extend `templates/base.html`.

**Design system** (`static/css/style.css`):
- Colors: deep teal `--bg: #06242b`, surface `#0c2f38`/`#133a44`, accent gold `--accent: #c2a378`
- Background: diagonal grid overlay + radial ambient glow via `body::before/::after`
- Navbar: glass morphism (`backdrop-filter: blur`), CSS-only hamburger for mobile
- Cards: hover with gold border glow + translateY(-4px)

**Pages:**
- `/` → landing page with hero, geometric SVG animation, 4 feature cards
- `/translator` → full-height chat UI (no footer, `translator-body` class for overflow:hidden)
- `/learning` → 6 level cards with gold numbered badges and progress bars
- `/streaming` → split layout: video panel left, scrolling transcript right

### External Dependencies

- **DeepSeek API** (`api.deepseek.com`): used for synonym extraction and final candidate judging. Two API calls per unmatched word.
- **Whisper** (local, `small` model): Arabic speech-to-text for voice input.
- **Sentence-transformers** (`intfloat/multilingual-e5-base`): embedding model for semantic search fallback.
- **ffmpeg**: video concatenation. All videos must share format (1920x1080, 25fps, H.264 High) for `-c copy` to work without lag.

### Video Format Constraint

All videos in `data/videos/` must be 1920x1080, 25fps, H.264 High profile. Letter videos (IDs 1-38) were batch-converted to match word videos. If new videos with different formats are added, they must be converted first or the concat will produce playback glitches.

### Known Issues / Future Work

- 76% of dictionary entries (5,989/7,841) have no synonyms — limits exact match quality for those entries.
- Synonym field is inconsistently formatted: some entries use comma-separated synonyms, most use space-separated multi-word phrases treated as single synonyms by `synonym_to_id` splitting logic.
- Embeddings index only encodes `wordAr`; including `description` would improve semantic search quality.
