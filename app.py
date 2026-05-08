import os
import re
import gradio as gr
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline

# =========================
# Configuración del modelo
# =========================
# Puedes cambiarlo si ya tienes otro modelo Instruct local.
MODEL_NAME = os.getenv("MODEL_NAME", "Qwen/Qwen2.5-0.5B-Instruct")

print(f"Cargando modelo: {MODEL_NAME} ...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    device_map="auto",
    trust_remote_code=True
)

llm = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
)

# =========================
# Utilidades
# =========================
def clean_text(text):
    if text is None:
        return ""
    text = re.sub(r"\s+", " ", text)
    return text.strip()

def split_sections(text):
    """
    Divide el texto de la SEC en bloques simples.
    Si encuentra encabezados tipo ITEM 1, ITEM 1A, etc., los separa.
    Si no, devuelve el texto completo como una sola sección.
    """
    text = text or ""
    pattern = r"(ITEM\s+\d+[A-Z]?(?:\.\d+)?)"
    matches = list(re.finditer(pattern, text, flags=re.IGNORECASE))

    if not matches:
        return [("Documento completo", text)]

    sections = []
    for i, match in enumerate(matches):
        start = match.start()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        title = match.group(1)
        section_text = text[start:end].strip()
        sections.append((title, section_text))

    return sections

def analyze_sentiment(text, mode="documento completo"):
    text = clean_text(text)

    if not text:
        return "No se recibió texto para analizar."

    # Limitar longitud para que corra rápido
    max_chars = 5000
    text_short = text[:max_chars]

    prompt = f"""
You are a financial sentiment analyst.
Analyze the sentiment of the following SEC filing text.
Return a short answer in Spanish with:
1. Sentimiento general: positivo, negativo o neutral
2. Breve explicación
3. Confidence from 1 to 5

Text:
{text_short}
"""

    try:
        output = llm(
            prompt,
            max_new_tokens=120,
            do_sample=False,
            temperature=0.1,
            return_full_text=False
        )[0]["generated_text"]

        return output.strip()
    except Exception as e:
        return f"Error al analizar el texto: {str(e)}"

def analyze_file(file_obj):
    if file_obj is None:
        return "Sube un archivo primero."

    path = file_obj.name
    ext = os.path.splitext(path)[1].lower()

    try:
        if ext in [".txt", ".md"]:
            with open(path, "r", encoding="utf-8", errors="ignore") as f:
                text = f.read()
        elif ext in [".html", ".htm"]:
            with open(path, "r", encoding="utf-8", errors="ignore") as f:
                text = f.read()
            # quitar etiquetas muy básico
            text = re.sub(r"<script.*?</script>", " ", text, flags=re.DOTALL | re.IGNORECASE)
            text = re.sub(r"<style.*?</style>", " ", text, flags=re.DOTALL | re.IGNORECASE)
            text = re.sub(r"<[^>]+>", " ", text)
        else:
            return "Formato no soportado. Usa .txt, .md, .html o .htm"

        return analyze_sentiment(text, mode="documento completo")

    except Exception as e:
        return f"Error leyendo archivo: {str(e)}"

def analyze_section(file_obj, section_name):
    if file_obj is None:
        return "Sube un archivo primero."

    path = file_obj.name
    ext = os.path.splitext(path)[1].lower()

    try:
        if ext in [".txt", ".md"]:
            with open(path, "r", encoding="utf-8", errors="ignore") as f:
                text = f.read()
        elif ext in [".html", ".htm"]:
            with open(path, "r", encoding="utf-8", errors="ignore") as f:
                text = f.read()
            text = re.sub(r"<script.*?</script>", " ", text, flags=re.DOTALL | re.IGNORECASE)
            text = re.sub(r"<style.*?</style>", " ", text, flags=re.DOTALL | re.IGNORECASE)
            text = re.sub(r"<[^>]+>", " ", text)
        else:
            return "Formato no soportado. Usa .txt, .md, .html o .htm"

        sections = split_sections(text)

        if section_name == "Todas las secciones":
            results = []
            for title, sec_text in sections[:10]:
                sentiment = analyze_sentiment(sec_text, mode=title)
                results.append(f"### {title}\n{sentiment}\n")
            return "\n".join(results)

        for title, sec_text in sections:
            if section_name.lower() in title.lower():
                return analyze_sentiment(sec_text, mode=title)

        return "No se encontró esa sección. Intenta con otra como ITEM 1, ITEM 1A, ITEM 7, etc."

    except Exception as e:
        return f"Error analizando sección: {str(e)}"


# =========================
# Interfaz Gradio
# =========================
with gr.Blocks(title="SEC Sentiment Analyzer") as demo:
    gr.Markdown("# SEC Sentiment Analyzer")
    gr.Markdown(
        "Sube un archivo de la SEC y obtén un análisis simple de sentimiento con un modelo Instruct local."
    )

    with gr.Row():
        file_input = gr.File(label="Subir archivo SEC", file_types=[".txt", ".md", ".html", ".htm"])
        section_input = gr.Textbox(
            label="Sección a analizar",
            placeholder="Ejemplo: ITEM 1A, ITEM 7 o escribe Todas las secciones",
            value="Documento completo"
        )

    with gr.Row():
        btn_doc = gr.Button("Analizar documento completo")
        btn_sec = gr.Button("Analizar sección")

    output = gr.Markdown(label="Resultado")

    btn_doc.click(fn=analyze_file, inputs=file_input, outputs=output)
    btn_sec.click(fn=analyze_section, inputs=[file_input, section_input], outputs=output)

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)
