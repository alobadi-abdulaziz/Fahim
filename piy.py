import json
import requests
import re
import subprocess
import os
import tempfile
import shutil
import whisper
import sounddevice as sd
import scipy.io.wavfile as wav
import numpy as np

# =========================
# CONFIG
# =========================
os.environ["PATH"] += r";C:\Users\ROIG1\Desktop\ffmpeg-8.1-essentials_build\bin"

FFMPEG         = r"C:\Users\ROIG1\Desktop\ffmpeg-8.1-essentials_build\bin\ffmpeg.exe"
VIDEOS_DIR     = r"C:\Users\ROIG1\Documents\مجلد جديد\data\videos"
API_KEY        = "sk-440e263c2de24986b7a81a6593e629dc"
URL            = "https://api.deepseek.com/v1/chat/completions"
RECORD_SECONDS = 5
SAMPLE_RATE    = 16000

# =========================
# 1) Load Dataset
# =========================
with open("sshi_metadata_try1.json", "r", encoding="utf-8") as f:
    data = json.load(f)

word_to_id    = {}
synonym_to_id = {}
id_to_path    = {}

for item in data:
    word_to_id[item["wordAr"]] = item["id"]
    id_to_path[item["id"]] = os.path.join(VIDEOS_DIR, item["video_filename"])

for item in data:
    synonyms_raw = item.get("synonym", "") or ""
    for syn in synonyms_raw.strip().split():
        syn = syn.strip()
        if syn and syn not in synonym_to_id:
            synonym_to_id[syn] = item["id"]


# =========================
# 2) Normalization
# =========================
def normalize(text):
    text = text.replace("أ", "ا")
    text = text.replace("إ", "ا")
    text = text.replace("آ", "ا")
    text = text.replace("ٱ", "ا")
    text = text.replace("ة", "ه")
    text = text.replace("ى", "ي")
    text = text.replace("ئ", "ي")
    text = re.sub(r'[\u064B-\u065F]', '', text)
    text = text.replace("ـ", "")
    text = text.strip()
    return text


# Build normalized lookups once at startup
normalized_word_map = {}
for original_word in word_to_id:
    norm = normalize(original_word)
    if norm not in normalized_word_map:
        normalized_word_map[norm] = original_word

normalized_synonym_map = {}
for syn in synonym_to_id:
    norm = normalize(syn)
    if norm not in normalized_synonym_map:
        normalized_synonym_map[norm] = syn


# =========================
# 3) Load Whisper Model
# =========================
print("Loading Whisper model...")
whisper_model = whisper.load_model("small")
print("Whisper ready.")


# =========================
# 4) Record & Transcribe
# =========================
def record_and_transcribe():
    print(f"\nRecording for {RECORD_SECONDS} seconds... speak now!")
    audio = sd.rec(
        int(RECORD_SECONDS * SAMPLE_RATE),
        samplerate=SAMPLE_RATE,
        channels=1,
        dtype="int16"
    )
    sd.wait()
    print("Recording done. Transcribing...")

    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        wav.write(tmp.name, SAMPLE_RATE, audio)
        tmp_path = tmp.name

    try:
        result = whisper_model.transcribe(tmp_path, language="ar")
        text = result["text"].strip()
        print(f"Transcribed: {text}")
        return text
    finally:
        os.unlink(tmp_path)


# =========================
# 5) DeepSeek API
# =========================
def get_keywords(text):
    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json"
    }

    prompt = f"""You are an Arabic dialect-to-Fusha keyword extractor for a sign language dictionary.

The dictionary only contains Modern Standard Arabic (Fusha) words.
Your job is to convert each meaningful word in the sentence to a list of possible Fusha equivalents.

Remove only non-essential words for sign language representation. Remove:
- Prepositions (في، من، على، إلى)
- Conjunctions (و، ثم، لكن، لأن، إذا)
- Fillers (اني، أني، يعني)

IMPORTANT:
- Do NOT remove words that affect meaning or intent
- Return ONLY meaningful sign language words
- Split the sentence into individual words (no phrases)
- For each word, provide up to 7 Fusha synonyms or equivalents
- Order synonyms from closest match to furthest (most likely to least likely to exist in a Fusha dictionary)
- Return ONLY valid JSON, no explanation

Format:
{{
  "words": [
    {{"original": "dialect_word", "fusha": ["closest1", "synonym2", "synonym3", "synonym4", "synonym5", "synonym6", "furthest7"]}},
    {{"original": "dialect_word", "fusha": ["fusha1"]}}
  ]
}}

Sentence:
{text}"""

    payload = {
        "model": "deepseek-chat",
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.1
    }

    try:
        response = requests.post(URL, headers=headers, json=payload, timeout=15)
        result = response.json()

        if "choices" not in result:
            print("API Error:", result)
            return []

        content = result["choices"][0]["message"]["content"].strip()

        if content.startswith("```"):
            content = content.split("```")[1]
            if content.startswith("json"):
                content = content[4:]
            content = content.strip()

        return json.loads(content)["words"]

    except requests.exceptions.Timeout:
        print("Error: API request timed out.")
        return []
    except json.JSONDecodeError:
        print("Error: Failed to parse API response.")
        return []
    except Exception as e:
        print(f"Unexpected error: {e}")
        return []


# =========================
# 6) Smart Search
# =========================
def find_id(word):
    if len(word) < 2:
        return None, None

    if word in word_to_id:
        return word_to_id[word], word

    norm_word = normalize(word)
    if norm_word in normalized_word_map:
        original = normalized_word_map[norm_word]
        return word_to_id[original], original

    if word in synonym_to_id:
        return synonym_to_id[word], word

    if norm_word in normalized_synonym_map:
        syn = normalized_synonym_map[norm_word]
        return synonym_to_id[syn], syn

    return None, None


def find_best_id(fusha_candidates):
    single_words = [c for c in fusha_candidates if " " not in c.strip()]
    phrases      = [c for c in fusha_candidates if " " in c.strip()]

    for candidate in single_words:
        result_id, result_word = find_id(candidate)
        if result_id is not None:
            return result_id, result_word

    for candidate in phrases:
        result_id, result_word = find_id(candidate)
        if result_id is not None:
            return result_id, result_word

    return None, None


# =========================
# 7) Merge Videos with ffmpeg
# =========================
def merge_videos(ids, output_path="output.mp4"):
    video_paths = []
    for vid_id in ids:
        path = id_to_path.get(vid_id)
        if path and os.path.exists(path):
            video_paths.append(path)
        else:
            print(f"  Warning: video not found for id {vid_id}")

    if not video_paths:
        print("No videos found to merge.")
        return None

    if len(video_paths) == 1:
        shutil.copy(video_paths[0], output_path)
        return output_path

    with tempfile.NamedTemporaryFile(mode="w", suffix=".txt",
                                     delete=False, encoding="utf-8") as f:
        for path in video_paths:
            f.write(f"file '{path}'\n")
        concat_file = f.name

    try:
        cmd = [
            FFMPEG, "-y",
            "-f", "concat",
            "-safe", "0",
            "-i", concat_file,
            "-c", "copy",
            output_path
        ]
        result = subprocess.run(cmd, capture_output=True, text=True)

        if result.returncode != 0:
            print("ffmpeg error:", result.stderr[-500:])
            return None

        return output_path

    finally:
        os.unlink(concat_file)


# =========================
# 8) Process Text → Videos
# =========================
def process(text):
    print("Processing...")
    words = get_keywords(text)

    if not words:
        print("No keywords extracted.")
        return

    ids = []
    matched_words = []
    not_found = []

    for entry in words:
        original   = entry.get("original", "")
        candidates = entry.get("fusha", [])

        print(f"  '{original}' → trying: {candidates}")

        result_id, result_word = find_best_id(candidates)

        if result_id is not None:
            ids.append(result_id)
            matched_words.append(result_word)
        else:
            not_found.append(original)

    print(f"\nMatched words : {matched_words}")
    print(f"Final IDs     : {ids}")

    if not_found:
        print(f"No match for  : {not_found}")

    if ids:
        print("\nMerging videos...")
        output = merge_videos(ids, output_path="output.mp4")
        if output:
            print(f"Done! Video saved to: {os.path.abspath(output)}")


# =========================
# 9) Main Loop
# =========================
print("\n" + "=" * 50)
print("Sign Language — Text & Voice Input")
print("Commands:")
print("  'voice' — record and transcribe")
print("  'exit'  — quit")
print("  anything else — type your sentence")
print("=" * 50)

while True:
    try:
        cmd = input("\n>> ").strip()
    except (KeyboardInterrupt, EOFError):
        print("\nExiting...")
        break

    if not cmd:
        continue

    if cmd.lower() == "exit":
        print("Exiting...")
        break

    if cmd.lower() == "voice":
        text = record_and_transcribe()
        if text:
            process(text)
    else:
        process(cmd)