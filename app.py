from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import os
import re

app = Flask(__name__)
CORS(app)

UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# ── helpers ──────────────────────────────────────────────────────────────────

def extract_sections(text: str) -> dict[str, str]:
    """Very lightweight SEC section detector."""
    section_patterns = {
        "Item 1 – Business": r"item\s+1[.\s]+business",
        "Item 1A – Risk Factors": r"item\s+1a[.\s]+risk factors",
        "Item 7 – MD&A": r"item\s+7[.\s]+management",
        "Item 8 – Financial Statements": r"item\s+8[.\s]+financial",
    }
    sections: dict[str, str] = {}
    lower = text.lower()
    positions = []
    for name, pattern in section_patterns.items():
        m = re.search(pattern, lower)
        if m:
            positions.append((m.start(), name))
    positions.sort()

    for i, (start, name) in enumerate(positions):
        end = positions[i + 1][0] if i + 1 < len(positions) else start + 3000
        snippet = text[start:end][:2500].strip()
        if snippet:
            sections[name] = snippet
    if not sections:
        sections["Full Document"] = text[:2500].strip()
    return sections


def run_sentiment(text: str) -> dict:
    """
    Call Qwen2-0.5B-Instruct via the transformers pipeline.
    Falls back gracefully if the model isn't installed yet.
    """
    prompt = (
        "You are a financial analyst. Analyze the sentiment of the following SEC filing excerpt. "
        "Reply ONLY with one of: Positive, Negative, or Neutral. Then on a new line, "
        "write one concise sentence explaining why.\n\n"
        f"Text:\n{text[:1200]}\n\nSentiment:"
    )

    try:
        from transformers import pipeline
        pipe = pipeline(
            "text-generation",
            model="Qwen/Qwen2-0.5B-Instruct",
            max_new_tokens=60,
            do_sample=False,
        )
        result = pipe(prompt)[0]["generated_text"]
        # Strip the prompt prefix
        answer = result[len(prompt):].strip()
        label_line = answer.split("\n")[0].strip()
        explanation = " ".join(answer.split("\n")[1:]).strip() if "\n" in answer else ""

        label = "Neutral"
        for candidate in ["Positive", "Negative", "Neutral"]:
            if candidate.lower() in label_line.lower():
                label = candidate
                break

        return {"label": label, "explanation": explanation or label_line, "model": "Qwen2-0.5B-Instruct"}

    except Exception as e:
        # Fallback: keyword heuristic so the UI still works without GPU/model
        pos_words = ["growth", "profit", "increase", "strong", "record", "expand", "revenue"]
        neg_words = ["loss", "risk", "decline", "litigation", "impairment", "restructur", "debt"]
        low = text.lower()
        pos = sum(low.count(w) for w in pos_words)
        neg = sum(low.count(w) for w in neg_words)
        label = "Positive" if pos > neg else ("Negative" if neg > pos else "Neutral")
        return {
            "label": label,
            "explanation": f"(Heuristic fallback – install transformers to use Qwen2) pos_signals={pos}, neg_signals={neg}",
            "model": "keyword-heuristic",
            "error": str(e),
        }

# ── routes ───────────────────────────────────────────────────────────────────

@app.route("/")
def index():
    return render_template("index.html")


@app.route("/analyze", methods=["POST"])
def analyze():
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files["file"]
    if file.filename == "":
        return jsonify({"error": "Empty filename"}), 400

    section_choice = request.form.get("section", "auto")

    # Read content
    raw = file.read()
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError:
        text = raw.decode("latin-1")

    sections = extract_sections(text)
    section_names = list(sections.keys())

    # Pick which section to analyse
    if section_choice == "auto" or section_choice not in sections:
        target_name = section_names[0]
    else:
        target_name = section_choice

    result = run_sentiment(sections[target_name])
    result["section_analyzed"] = target_name
    result["available_sections"] = section_names
    result["char_count"] = len(text)

    return jsonify(result)


@app.route("/sections", methods=["POST"])
def get_sections():
    if "file" not in request.files:
        return jsonify({"error": "No file"}), 400
    file = request.files["file"]
    raw = file.read()
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError:
        text = raw.decode("latin-1")
    sections = extract_sections(text)
    return jsonify({"sections": list(sections.keys())})


if __name__ == "__main__":
    app.run(debug=True, port=5000)