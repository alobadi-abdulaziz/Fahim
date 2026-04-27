"""
app.py
------
Flask web server — connects search.py + whisper + ffmpeg to the UI.
Run: python app.py
Open: http://localhost:5000
"""

from flask import Flask, request, jsonify, send_file, render_template
import os
import tempfile
import shutil
import subprocess
import whisper
import sounddevice as sd
import scipy.io.wavfile as wav

from search import translate_sentence, id_to_path

# =========================
# CONFIG
# =========================
os.environ["PATH"] += r";C:\Users\oscar\Desktop\ffmpeg-8.1-essentials_build\bin"
FFMPEG         = r"C:\Users\oscar\Desktop\ffmpeg-8.1-essentials_build\bin\ffmpeg.exe"
OUTPUT_VIDEO   = "output.mp4"
RECORD_SECONDS = 5
SAMPLE_RATE    = 16000

app = Flask(__name__)

# =========================
# Load Whisper
# =========================
print("Loading Whisper model...")
whisper_model = whisper.load_model("small")
print("Whisper ready.")

# =========================
# Merge Videos
# =========================
def merge_videos(ids):
    video_paths = []
    for vid_id in ids:
        path = id_to_path.get(vid_id)
        if path and os.path.exists(path):
            video_paths.append(path)
        else:
            print(f"  Warning: video not found for id {vid_id}")

    if not video_paths:
        return None

    if len(video_paths) == 1:
        shutil.copy(video_paths[0], OUTPUT_VIDEO)
        return OUTPUT_VIDEO

    with tempfile.NamedTemporaryFile(mode="w", suffix=".txt",
                                     delete=False, encoding="utf-8") as f:
        for path in video_paths:
            f.write(f"file '{path}'\n")
        concat_file = f.name

    try:
        cmd = [FFMPEG, "-y", "-f", "concat", "-safe", "0",
               "-i", concat_file, "-c", "copy", OUTPUT_VIDEO]
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            print("ffmpeg error:", result.stderr[-300:])
            return None
        return OUTPUT_VIDEO
    finally:
        os.unlink(concat_file)

# =========================
# Routes
# =========================
@app.route("/")
def index():
    return render_template("index.html")

@app.route("/translator")
def translator():
    return render_template("translator.html", active="translator")

@app.route("/learning")
def learning():
    return render_template("learning.html", active="learning")

@app.route("/streaming")
def streaming():
    return render_template("streaming.html", active="streaming")

@app.route("/chat", methods=["POST"])
def chat():
    body = request.get_json()
    text = (body.get("text") or "").strip()
    if not text:
        return jsonify({"error": "empty input"}), 400

    results, not_found = translate_sentence(text)

    if not results:
        return jsonify({"error": "no matches found", "not_found": not_found}), 404

    ids     = [r["id"]   for r in results]
    matched = [r["word"] for r in results]

    output = merge_videos(ids)
    if not output:
        return jsonify({"error": "video merge failed"}), 500

    return jsonify({
        "matched":   matched,
        "ids":       ids,
        "not_found": not_found,
        "video_url": "/video"
    })

@app.route("/voice", methods=["POST"])
def voice():
    print("Recording...")
    audio = sd.rec(int(RECORD_SECONDS * SAMPLE_RATE),
                   samplerate=SAMPLE_RATE, channels=1, dtype="int16")
    sd.wait()

    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        wav.write(tmp.name, SAMPLE_RATE, audio)
        tmp_path = tmp.name

    try:
        result = whisper_model.transcribe(tmp_path, language="ar")
        return jsonify({"text": result["text"].strip()})
    except Exception as e:
        return jsonify({"error": str(e)}), 500
    finally:
        os.unlink(tmp_path)

@app.route("/video")
def video():
    if not os.path.exists(OUTPUT_VIDEO):
        return jsonify({"error": "no video"}), 404
    return send_file(OUTPUT_VIDEO, mimetype="video/mp4")

if __name__ == "__main__":
    app.run(debug=False, port=5000)