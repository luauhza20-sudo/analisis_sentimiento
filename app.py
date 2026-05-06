from flask import Flask, request, render_template, jsonify
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import PyPDF2
from bs4 import BeautifulSoup
import io
import re

app = Flask(__name__)

print("Cargando modelo Qwen2...")
model_name = "Qwen/Qwen2-0.5B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.float32,
    device_map="auto"
)
print("Modelo listo!")

def extract_text(file, filename):
    content = file.read()
    if filename.endswith(".pdf"):
        reader = PyPDF2.PdfReader(io.BytesIO(content))
        return "\n".join(page.extract_text() or "" for page in reader.pages)
    elif filename.endswith((".htm", ".html")):
        soup = BeautifulSoup(content, "html.parser")
        for tag in soup(["script", "style"]):
            tag.decompose()
        return soup.get_text(separator="\n")
    else:
        return content.decode("utf-8", errors="ignore")

def split_sections(text):
    section_patterns = {
        "Full Document": None,
        "Item 1 - Business": r"item\s*1[^a-z]*business",
        "Item 1A - Risk Factors": r"item\s*1a[^a-z]*risk\s*factor",
        "Item 7 - MD&A": r"item\s*7[^a-z]*(management|discussion)",
        "Item 8 - Financial Statements": r"item\s*8[^a-z]*financial",
    }
    sections = {"Full Document": text}
    lines = text.split("\n")
    current = "Full Document"
    buffer = {"Full Document": []}

    for line in lines:
        for name, pattern in section_patterns.items():
            if pattern and re.search(pattern, line.lower()):
                current = name
                buffer[current] = []
                break
        buffer.setdefault(current, []).append(line)

    return {k: "\n".join(v).strip() for k, v in buffer.items() if len("\n".join(v).strip()) > 100}

def analyze(text):
    truncated = text[:1800]
    messages = [
        {
            "role": "system",
            "content": (
                "You are a financial analyst specialized in SEC filings. "
                "Analyze the sentiment and respond with:\n"
                "**Sentiment:** [Positive/Negative/Neutral/Mixed]\n"
                "**Confidence:** [High/Medium/Low]\n"
                "**Key Signals:** [bullet points]\n"
                "**Summary:** [2-3 sentences]"
            )
        },
        {
            "role": "user",
            "content": f"Analyze the sentiment of this SEC filing:\n\n{truncated}"
        }
    ]
    formatted = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer([formatted], return_tensors="pt").to(model.device)
    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=300,
            temperature=0.7,
            top_p=0.9,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id
        )
    new_tokens = output_ids[0][inputs.input_ids.shape[1]:]
    return tokenizer.decode(new_tokens, skip_special_tokens=True)

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/upload", methods=["POST"])
def upload():
    file = request.files.get("file")
    if not file:
        return jsonify({"error": "No file uploaded"}), 400
    text = extract_text(file, file.filename)
    sections = split_sections(text)
    return jsonify({
        "sections": list(sections.keys()),
        "texts": sections
    })

@app.route("/analyze", methods=["POST"])
def analyze_route():
    data = request.json
    text = data.get("text", "")
    if not text:
        return jsonify({"error": "No text provided"}), 400
    result = analyze(text)
    return jsonify({"result": result})

import os
port = int(os.environ.get("PORT", 5000))
app.run(host="0.0.0.0", port=port)
