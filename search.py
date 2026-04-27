"""
search.py
---------
Three-layer search engine for Arabic Sign Language:
  1. Exact match  (normalize + strip ال)
  2. Embeddings   (original word + all fusha candidates combined)
  3. DeepSeek     (picks best ID from all candidates with full sentence context)
"""

import json
import os
import re
import requests
import threading
import numpy as np
from sentence_transformers import SentenceTransformer

# =========================
# CONFIG
# =========================
API_KEY      = "sk-440e263c2de24986b7a81a6593e629dc"
DEEPSEEK_URL = "https://api.deepseek.com/v1/chat/completions"
VIDEOS_DIR   = r"C:\Users\oscar\Desktop\Sing_Lun\data\videos"
INDEX_FILE   = "embeddings_index.npz"
MODEL_NAME   = "intfloat/multilingual-e5-base"

os.environ["PATH"] += r";C:\Users\oscar\Desktop\ffmpeg-8.1-essentials_build\bin"

# =========================
# 1) Load Dataset
# =========================
with open("sshi_metadata_try1.json", "r", encoding="utf-8") as f:
    raw_data = json.load(f)

all_words = [item["wordAr"] for item in raw_data]
all_ids   = [item["id"]     for item in raw_data]

word_to_id = {item["wordAr"]: item["id"] for item in raw_data}
id_to_path = {item["id"]: os.path.join(VIDEOS_DIR, item["video_filename"]) for item in raw_data}
id_to_word = {item["id"]: item["wordAr"] for item in raw_data}
id_to_synonym     = {item["id"]: (item.get("synonym") or "") for item in raw_data}
id_to_description = {item["id"]: (item.get("description") or "") for item in raw_data}

synonym_to_id = {}
for item in raw_data:
    for syn in (item.get("synonym") or "").split():
        syn = syn.strip()
        if syn and syn not in synonym_to_id:
            synonym_to_id[syn] = item["id"]

# =========================
# 2) Normalization — يشيل ال التعريف + همزات + تشكيل
# =========================
def normalize(text: str) -> str:
    text = text.strip()
    # شيل ال التعريف من البداية
    if text.startswith("ال") and len(text) > 3:
        text = text[2:]
    # توحيد الهمزات
    text = text.replace("أ","ا").replace("إ","ا").replace("آ","ا").replace("ٱ","ا")
    # توحيد التاء المربوطة والألف المقصورة
    text = text.replace("ة","ه").replace("ى","ي").replace("ئ","ي")
    # شيل التشكيل
    text = re.sub(r'[\u064B-\u065F]', '', text)
    # شيل التطويل
    text = text.replace("ـ","")
    return text.strip()

# بناء lookup tables مرة وحدة عند التشغيل
normalized_word_map = {}
for w in word_to_id:
    n = normalize(w)
    if n not in normalized_word_map:
        normalized_word_map[n] = w

normalized_synonym_map = {}
for s in synonym_to_id:
    n = normalize(s)
    if n not in normalized_synonym_map:
        normalized_synonym_map[n] = s

# =========================
# 3) Embeddings Index
# =========================
print("Loading embedding model...")
emb_model = SentenceTransformer(MODEL_NAME)

if os.path.exists(INDEX_FILE):
    print("Loading saved embeddings index...")
    saved      = np.load(INDEX_FILE, allow_pickle=True)
    embeddings = saved["embeddings"]
else:
    print(f"Building embeddings index for {len(all_words)} words (first time only)...")
    passage_words = [f"passage: {w}" for w in all_words]
    embeddings = emb_model.encode(passage_words, show_progress_bar=True, batch_size=64)
    np.savez(INDEX_FILE, embeddings=embeddings)
    print("Index saved.")

print("Search engine ready.\n")

# =========================
# 4) Exact Search
# =========================
def exact_search(word: str):
    """Returns (id, matched_word) or (None, None)"""
    if not word or len(word) < 2:
        return None, None
    # 1. عين بعين
    if word in word_to_id:
        return word_to_id[word], word
    # 2. بعد normalize (يشيل ال + همزات)
    nw = normalize(word)
    if nw in normalized_word_map:
        orig = normalized_word_map[nw]
        return word_to_id[orig], orig
    # 3. synonym عين بعين
    if word in synonym_to_id:
        return synonym_to_id[word], word
    # 4. synonym بعد normalize
    if nw in normalized_synonym_map:
        syn = normalized_synonym_map[nw]
        return synonym_to_id[syn], syn
    return None, None

# =========================
# 5) Embedding Search — يأخذ قائمة queries ويجمع النتائج
# =========================
def embedding_search(queries: list, top_k: int = 5):
    """
    queries: list of strings (original word + all fusha candidates)
    Returns deduplicated list of (id, word, score) sorted by best score
    """
    if not queries:
        return []

    best_per_id = {}  # id → (word, score)

    for q in queries:
        if not q or len(q.strip()) < 2:
            continue
        qvec   = emb_model.encode([f"query: {q}"])[0]
        norms  = np.linalg.norm(embeddings, axis=1) * np.linalg.norm(qvec)
        scores = np.dot(embeddings, qvec) / (norms + 1e-9)
        top_i  = np.argsort(scores)[::-1][:top_k]
        for i in top_i:
            cid   = all_ids[i]
            score = float(scores[i])
            if cid not in best_per_id or best_per_id[cid][1] < score:
                best_per_id[cid] = (all_words[i], score)

    sorted_res = sorted(best_per_id.items(), key=lambda x: x[1][1], reverse=True)
    return [(cid, word, round(score, 4)) for cid, (word, score) in sorted_res[:top_k * 2]]

# =========================
# 6) DeepSeek — Get Synonyms
# =========================
def deepseek_synonyms(sentence: str):
    """
    Returns list of {"original": str, "fusha": [str, ...]}
    """
    headers = {"Authorization": f"Bearer {API_KEY}", "Content-Type": "application/json"}
    prompt  = f"""You are an Arabic dialect-to-Fusha keyword extractor for a sign language dictionary.

The dictionary contains Modern Standard Arabic (Fusha) words AND full phrases.
Convert each meaningful word in the sentence to up to 7 Fusha equivalents.

Rules:
- Remove ONLY: prepositions (في، من، على، إلى), conjunctions (و، ثم، لكن، لأن، إذا), fillers (اني، أني، يعني)
- Do NOT remove words that affect meaning or intent
- Output individual words only (no multi-word phrases per entry)
- Order synonyms: closest match first, furthest last
- NAMES RULE: If a word is a proper name (person name, place name, brand, scientific name) that has no direct Arabic meaning, mark it as a name and split it into its individual Arabic letters. Examples: نيوتن → ["ن","ي","و","ت","ن"], محمد → ["م","ح","م","د"]
- Return ONLY valid JSON, no explanation

Format for normal words:
{{"original": "w", "fusha": ["f1","f2","f3","f4","f5","f6","f7"]}}

Format for names:
{{"original": "name", "is_name": true, "letters": ["letter1","letter2","letter3"]}}

Full format:
{{"words": [...]}}

Sentence: {sentence}"""

    payload = {"model": "deepseek-chat",
               "messages": [{"role": "user", "content": prompt}],
               "temperature": 0.1}
    try:
        res     = requests.post(DEEPSEEK_URL, headers=headers, json=payload, timeout=15)
        content = res.json()["choices"][0]["message"]["content"].strip()
        if content.startswith("```"):
            content = content.split("```")[1]
            if content.startswith("json"):
                content = content[4:]
            content = content.strip()
        return json.loads(content)["words"]
    except Exception as e:
        print(f"  [DeepSeek synonyms error] {e}")
        return []

# =========================
# 7) DeepSeek — Final Judge
# =========================
def deepseek_judge(original_word: str, sentence: str, candidates: list):
    """
    candidates: [{"id": int, "word": str, "source": str}]
    Returns best_id (int) or None
    """
    if not candidates:
        return None

    headers = {"Authorization": f"Bearer {API_KEY}", "Content-Type": "application/json"}

    blocks = []
    for c in candidates:
        syn  = (id_to_synonym.get(c["id"]) or "").strip()
        desc = (id_to_description.get(c["id"]) or "").strip()
        block = f"  ID {c['id']}: {c['word']}  [{c['source']}]"
        if syn and syn.lower() != "null":
            block += f"\n     synonyms:    {syn[:200]}"
        if desc and desc.lower() != "null":
            block += f"\n     description: {desc[:200]}"
        blocks.append(block)
    options = "\n".join(blocks)

    prompt = f"""You are a sign language dictionary selector.

Full sentence context: "{sentence}"
Word to translate: "{original_word}"

Candidate entries from the dictionary. Each entry shows the dictionary word, optional synonyms, and a description of what the sign actually means:
{options}

Task:
- Pick the single best ID for "{original_word}" given the full sentence context.
- USE THE DESCRIPTION to verify the entry's true meaning before choosing it.
- REJECT any candidate whose description does NOT match the meaning of "{original_word}". For example, if the query is a question word like "ماهو" and a candidate's description describes an emotion or an unrelated phrase, do NOT pick it.
- Prefer simple, direct matches over long phrase entries unless the phrase clearly matches the query word.
- Full phrase entries are valid only if they truly match the query meaning.
- Return ONLY valid JSON, no explanation.

Format:
{{"best_id": <number>}}"""

    payload = {"model": "deepseek-chat",
               "messages": [{"role": "user", "content": prompt}],
               "temperature": 0.0}
    try:
        res     = requests.post(DEEPSEEK_URL, headers=headers, json=payload, timeout=15)
        content = res.json()["choices"][0]["message"]["content"].strip()
        if content.startswith("```"):
            content = content.split("```")[1]
            if content.startswith("json"):
                content = content[4:]
            content = content.strip()
        return json.loads(content)["best_id"]
    except Exception as e:
        print(f"  [DeepSeek judge error] {e}")
        return None

# =========================
# 8) Per-Word Pipeline
# =========================
def search_word(original_word: str, fusha_candidates: list, sentence: str):
    """
    Returns {"id": int, "word": str} or None

    Step A: exact match on every fusha candidate
    Step B: embedding search using original + ALL fusha candidates as queries
    Step C: DeepSeek judges all collected candidates with full sentence context
    """
    collected = {}  # id → {"id", "word", "source"}
    lock = threading.Lock()

    def add(cid, cword, source):
        if cid:
            with lock:
                if cid not in collected:
                    collected[cid] = {"id": cid, "word": cword, "source": source}

    # ── Prepare embedding queries for Step B ─────────────────────
    original_clean = normalize(original_word)
    embed_queries  = [original_word, original_clean] + fusha_candidates
    seen_q = set()
    unique_queries = []
    for q in embed_queries:
        if q and q not in seen_q:
            seen_q.add(q)
            unique_queries.append(q)

    # ── Run Step A and Step B in parallel ─────────────────────────
    def step_a():
        for f in fusha_candidates:
            cid, cword = exact_search(f)
            if cid:
                add(cid, cword, "exact")

    def step_b():
        for cid, cword, score in embedding_search(unique_queries, top_k=5):
            add(cid, cword, f"embed:{score:.3f}")

    thread_b = threading.Thread(target=step_b)
    thread_b.start()
    step_a()
    thread_b.join()

    if not collected:
        return None

    candidates_list = list(collected.values())

    # لو في نتيجة وحدة فارجعها مباشرة
    if len(candidates_list) == 1:
        return {"id": candidates_list[0]["id"], "word": candidates_list[0]["word"]}

    # ── Step C: DeepSeek Judge ────────────────────────────────────
    print(f"    → Judge for '{original_word}': {[c['word'] for c in candidates_list]}")
    best_id = deepseek_judge(original_word, sentence, candidates_list)

    if best_id and best_id in collected:
        return {"id": best_id, "word": collected[best_id]["word"]}

    # Fallback: exact أولاً، ثم embed
    exact_hits = [c for c in candidates_list if c["source"] == "exact"]
    if exact_hits:
        return {"id": exact_hits[0]["id"], "word": exact_hits[0]["word"]}
    return {"id": candidates_list[0]["id"], "word": candidates_list[0]["word"]}


# =========================
# 9) Phrase-First Match
# =========================
def phrase_match_spans(sentence: str, max_n: int = 5):
    """
    Greedy longest-first n-gram match against the dictionary (sizes max_n..2).
    Exact match only — single-word matches go through the per-word pipeline.
    Returns: dict {start_idx: (end_idx, id, word)}.
    """
    tokens = sentence.split()
    n      = len(tokens)
    locked = [False] * n
    matches = {}

    for size in range(min(n, max_n), 1, -1):
        for start in range(n - size + 1):
            end = start + size
            if any(locked[i] for i in range(start, end)):
                continue
            phrase = " ".join(tokens[start:end])
            cid, cword = exact_search(phrase)
            if cid:
                matches[start] = (end, cid, cword)
                for i in range(start, end):
                    locked[i] = True

    return matches

# =========================
# 10) Full Sentence Pipeline
# =========================
def translate_sentence(sentence: str):
    """
    Returns ([{"id", "word"}, ...], [not_found_words])

    Pipeline:
      1. Phrase-first: try multi-word dictionary entries against n-grams of the sentence.
         Locked spans skip the per-word pipeline.
      2. Remaining gaps → DeepSeek splits into words → exact / embeddings / judge.
    """
    print(f"\nTranslating: '{sentence}'")

    tokens     = sentence.split()
    n          = len(tokens)
    phrase_at  = phrase_match_spans(sentence)

    results    = []
    not_found  = []
    gap_tokens = []

    def flush_gap():
        if not gap_tokens:
            return
        gap_str = " ".join(gap_tokens)
        word_entries = deepseek_synonyms(gap_str)
        for entry in word_entries:
            original = entry.get("original", "")

            if entry.get("is_name"):
                letters = entry.get("letters", [])
                print(f"  '{original}' → fingerspell: {letters}")
                for letter in letters:
                    cid = word_to_id.get(letter)
                    if not cid:
                        nl = normalize(letter)
                        orig = normalized_word_map.get(nl) or normalized_word_map.get(letter)
                        if orig:
                            cid = word_to_id.get(orig)
                    if cid:
                        results.append({"id": cid, "word": id_to_word.get(cid, letter)})
                    else:
                        print(f"    ✗ letter '{letter}' not found")
                continue

            candidates = entry.get("fusha", [])
            print(f"  '{original}' → fusha candidates: {candidates}")
            result = search_word(original, candidates, sentence)
            if result:
                print(f"    ✓ {result['word']} (ID {result['id']})")
                results.append(result)
            else:
                print(f"    ✗ no match")
                not_found.append(original)
        gap_tokens.clear()

    i = 0
    while i < n:
        if i in phrase_at:
            flush_gap()
            end, cid, cword = phrase_at[i]
            print(f"  ✓ phrase match: '{cword}' (ID {cid})")
            results.append({"id": cid, "word": cword})
            i = end
        else:
            gap_tokens.append(tokens[i])
            i += 1
    flush_gap()

    return results, not_found