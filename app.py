import gradio as gr
import re
import json
import pandas as pd
from collections import defaultdict
from langdetect import detect
from transformers import pipeline
import tempfile

# ─────────────────────────────────────────────
# AI MODEL
# ─────────────────────────────────────────────

MODEL_NAME = "mistralai/Mistral-7B-Instruct-v0.2"

try:
    print(f"⏳ Cargando modelo {MODEL_NAME}…")
    generator = pipeline(
        "text-generation",
        model=MODEL_NAME,
        device_map="auto",
        torch_dtype="auto",
    )
    MODEL_LOADED = True
    print("✅ Modelo cargado correctamente")
except Exception as _model_err:
    print(f"❌ No se pudo cargar el modelo: {_model_err}")
    generator = None
    MODEL_LOADED = False

# ─────────────────────────────────────────────
# PROMPT BUILDER
# ─────────────────────────────────────────────

def build_prompt(text: str, lang: str) -> str:
    """Builds the generation prompt based on detected language."""
    if lang == "es":
        return (
            "Crea un acta estructurada en Markdown con estas secciones exactas:\n\n"
            "## Participantes\n"
            "## Temas tratados\n"
            "## Decisiones tomadas\n"
            "## Acciones y responsables\n"
            "## Próximos pasos\n\n"
            "Instrucciones:\n"
            "- No inventes información que no esté en el texto.\n"
            "- Sé conciso y claro.\n"
            "- Si no hay información para una sección, escribe 'Sin información'.\n\n"
            f"Texto de la reunión:\n{text}\n\nActa:\n"
        )
    return (
        "Create structured meeting minutes in Markdown with these exact sections:\n\n"
        "## Participants\n"
        "## Topics Discussed\n"
        "## Decisions Made\n"
        "## Action Items and Owners\n"
        "## Next Steps\n\n"
        "Instructions:\n"
        "- Do not invent information not present in the text.\n"
        "- Be concise and clear.\n"
        "- If there is no information for a section, write 'No information'.\n\n"
        f"Meeting text:\n{text}\n\nMinutes:\n"
    )

# ─────────────────────────────────────────────
# REGEX PATTERNS  (compiled at module level)
# ─────────────────────────────────────────────

_TASK_VERBS = re.compile(
    r"\b("
    r"corregir|revisar|actualizar|validar|configurar|mejorar|documentar|redactar|coordinar|"
    r"crear|enviar|preparar|verificar|analizar|completar|implementar|definir|"
    r"update|review|fix|validate|configure|improve|document|draft|coordinate|"
    r"implement|send|share|prepare|check|complete|define"
    r")\b",
    re.IGNORECASE,
)

_RESPONSIBLE_PATTERNS = [
    re.compile(r"Responsable:\s*([A-Za-zÁÉÍÓÚÑáéíóúñ ]+?)(?:[.,;]|$)", re.IGNORECASE),
    re.compile(r"Asignado a:?\s*([A-Za-zÁÉÍÓÚÑáéíóúñ ]+?)(?:[.,;]|$)", re.IGNORECASE),
    re.compile(r"([A-Za-zÁÉÍÓÚÑáéíóúñ ]+?)\s+se encarga de", re.IGNORECASE),
    re.compile(r"Responsible:\s*([A-Za-zÁÉÍÓÚÑáéíóúñ ]+?)(?:[.,;]|$)", re.IGNORECASE),
    re.compile(r"Assigned to:?\s*([A-Za-zÁÉÍÓÚÑáéíóúñ ]+?)(?:[.,;]|$)", re.IGNORECASE),
    re.compile(r"([A-Za-zÁÉÍÓÚÑáéíóúñ ]+?)\s+is responsible for", re.IGNORECASE),
    re.compile(r"Owner:\s*([A-Za-zÁÉÍÓÚÑáéíóúñ ]+?)(?:[.,;]|$)", re.IGNORECASE),
]

_DATE_PATTERN = re.compile(
    r"\b(\d{4}-\d{2}-\d{2}|\d{2}/\d{2}/\d{4}|\d{2}-\d{2}-\d{4})\b"
)

_PARTICIPANT_PATTERN = re.compile(r"^\*?\s*([A-Za-zÁÉÍÓÚÑáéíóúñ]+):")

_DECISION_PATTERN = re.compile(
    r"\b("
    r"se acuerda|se decide|priorizar|quedamos en|se establece|se define|se sugiere|"
    r"se aprueba|se confirma|acordamos|"
    r"we agree|we decide|prioritize|is agreed|it is decided|is recommended|"
    r"we suggest|it was decided|it was agreed|team agreed|the team will"
    r")\b",
    re.IGNORECASE,
)

# Words that should NOT count as participant names
_IGNORE_NAMES: set[str] = {
    "Fecha", "Hora", "Lugar", "Agenda", "Notas", "Reunión", "Acta", "Tema",
    "Date", "Time", "Location", "Notes", "Tasks", "Decisions", "Meeting", "Minutes",
    "Participantes", "Participants", "Temas", "Topics", "Próximos", "Next",
    "Actions", "Resumen", "Summary", "Decisiones", "Objetivo", "Objective",
    "Tareas", "Acciones",
}

# Phrases that disqualify a line as a task (too vague / conversational)
_TASK_SKIP_PHRASES = [
    "propongo", "de acuerdo", "vamos a", "se sugiere",
    "we suggest", "we agree", "let's", "i think", "maybe", "perhaps",
    "podríamos", "deberíamos", "tal vez",
]

# ─────────────────────────────────────────────
# REGEX PARSER
# ─────────────────────────────────────────────

def parse_meeting_minutes(text: str) -> dict:
    """
    Parses meeting-minutes text with regex.
    Returns a dict with keys: tasks, decisions, participation.
    """
    tasks: list[dict] = []
    decisions: list[str] = []
    participation: defaultdict[str, int] = defaultdict(int)

    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line:
            continue

        # ── Participation ─────────────────────────
        m = _PARTICIPANT_PATTERN.match(line)
        if m:
            name = m.group(1).strip()
            if name not in _IGNORE_NAMES:
                participation[name] += 1

        # ── Decisions (priority: processed first) ─
        if _DECISION_PATTERN.search(line):
            cleaned = re.sub(r"^[A-Za-zÁÉÍÓÚÑáéíóúñ]+:\s*", "", line).strip()
            # Exclude lines that are really task assignments ("se decide revisar…")
            if cleaned and not _TASK_VERBS.search(cleaned):
                decisions.append(cleaned)
            continue  # skip task check for this line

        # ── Tasks ─────────────────────────────────
        if _TASK_VERBS.search(line):
            line_lower = line.lower()
            if any(phrase in line_lower for phrase in _TASK_SKIP_PHRASES):
                continue

            # Extract responsible
            responsible = "N/A"
            for pat in _RESPONSIBLE_PATTERNS:
                hit = pat.search(line)
                if hit:
                    responsible = hit.group(1).strip()
                    break

            # Extract date
            date_hit = _DATE_PATTERN.search(line)
            due_date = date_hit.group(0) if date_hit else "N/A"

            # Clean task text: remove metadata suffixes
            task_text = re.sub(r"(Responsable:|Responsible:|Owner:).*$", "", line).strip()
            task_text = re.sub(r"[–-]\s*\d{4}-\d{2}-\d{2}", "", task_text).strip()
            task_text = re.sub(r"\s{2,}", " ", task_text).strip()

            if task_text:
                tasks.append({
                    "task": task_text,
                    "responsible": responsible,
                    "due_date": due_date,
                })

    return {
        "tasks": tasks,
        "decisions": decisions,
        "participation": dict(participation),
    }

# ─────────────────────────────────────────────
# EXPORT HELPERS
# ─────────────────────────────────────────────

def insights_to_markdown(insights: dict) -> str:
    md = "# 📊 Meeting Insights\n\n"

    md += "## 👥 Participación / Participation\n"
    if insights["participation"]:
        for person, count in sorted(insights["participation"].items(), key=lambda x: -x[1]):
            bar = "█" * count
            md += f"- **{person}**: {count} intervención(es) {bar}\n"
    else:
        md += "_No se detectaron participantes._\n"

    md += "\n## ✅ Tareas / Tasks\n"
    if insights["tasks"]:
        for i, t in enumerate(insights["tasks"], 1):
            md += (
                f"{i}. **{t['task']}**\n"
                f"   - Responsable: `{t['responsible']}`\n"
                f"   - Fecha / Due date: `{t['due_date']}`\n"
            )
    else:
        md += "_No se detectaron tareas._\n"

    md += "\n## 🔑 Decisiones / Decisions\n"
    if insights["decisions"]:
        for d in insights["decisions"]:
            md += f"- {d}\n"
    else:
        md += "_No se detectaron decisiones._\n"

    return md


def _save_temp(content: str, suffix: str) -> str:
    """Write content to a temp file and return its path."""
    with tempfile.NamedTemporaryFile(
        delete=False, suffix=suffix, mode="w", encoding="utf-8"
    ) as tmp:
        tmp.write(content)
        return tmp.name


def _build_export_files(insights: dict) -> tuple[str, str]:
    """Return paths for a .md and .json export of the insights."""
    md_path = _save_temp(insights_to_markdown(insights), ".md")
    json_path = _save_temp(
        json.dumps(insights, indent=2, ensure_ascii=False), ".json"
    )
    return md_path, json_path

# ─────────────────────────────────────────────
# CORE LOGIC
# ─────────────────────────────────────────────

def generate_minutes(meeting_text: str) -> tuple[str, str]:
    """
    Generate structured meeting minutes via Mistral-7B.
    Returns (markdown_output, status_message).
    """
    if not meeting_text.strip():
        return "⚠️ Por favor pega el texto de la reunión.", "⚠️ Sin texto de entrada"

    if not MODEL_LOADED or generator is None:
        return (
            "❌ Modelo no disponible.\n"
            "Verifica que `transformers`, `torch` y el modelo estén instalados.",
            "❌ Modelo no cargado",
        )

    try:
        lang = detect(meeting_text)
    except Exception:
        lang = "es"

    prompt = build_prompt(meeting_text, lang)

    try:
        output = generator(
            prompt,
            max_new_tokens=700,
            temperature=0,
            do_sample=False,
        )
        result = output[0]["generated_text"].replace(prompt, "").strip()
        status = f"✅ Acta generada | Idioma detectado: {lang.upper()}"
        return result, status
    except Exception as exc:
        return f"❌ Error durante la generación:\n{exc}", "❌ Error en generación"


def analyze_minutes(
    minutes_text: str,
    file_input=None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, str, str | None, str | None]:
    """
    Analyze meeting minutes with regex.
    Accepts text or an uploaded .txt/.md file.
    Returns (participation_df, tasks_df, decisions_df, status, md_path, json_path).
    """
    _empty = pd.DataFrame()

    # File input takes priority over pasted text
    if file_input is not None:
        try:
            # Handle Gradio 3.x (object with .name) and 4.x (string path or dict)
            if isinstance(file_input, dict):
                file_path = file_input.get("name") or file_input.get("path", "")
            elif isinstance(file_input, str):
                file_path = file_input
            else:
                file_path = getattr(file_input, "name", str(file_input))

            with open(file_path, "r", encoding="utf-8") as f:
                minutes_text = f.read()
        except Exception as exc:
            return _empty, _empty, _empty, f"❌ Error leyendo archivo: {exc}", None, None

    if not minutes_text or not minutes_text.strip():
        return (
            _empty, _empty, _empty,
            "⚠️ Ingresa texto o sube un archivo .txt / .md",
            None, None,
        )

    insights = parse_meeting_minutes(minutes_text)

    # Build DataFrames
    participation_df = (
        pd.DataFrame(
            sorted(insights["participation"].items(), key=lambda x: -x[1]),
            columns=["Persona / Person", "Intervenciones"],
        )
        if insights["participation"]
        else pd.DataFrame(columns=["Persona / Person", "Intervenciones"])
    )

    tasks_df = (
        pd.DataFrame(
            [
                {
                    "Tarea / Task": t["task"],
                    "Responsable": t["responsible"],
                    "Fecha / Due Date": t["due_date"],
                }
                for t in insights["tasks"]
            ]
        )
        if insights["tasks"]
        else pd.DataFrame(columns=["Tarea / Task", "Responsable", "Fecha / Due Date"])
    )

    decisions_df = (
        pd.DataFrame({"Decisión / Decision": insights["decisions"]})
        if insights["decisions"]
        else pd.DataFrame(columns=["Decisión / Decision"])
    )

    md_path, json_path = _build_export_files(insights)

    n_p = len(insights["participation"])
    n_t = len(insights["tasks"])
    n_d = len(insights["decisions"])
    status = (
        f"✅ Análisis completado | "
        f"👥 {n_p} participantes | "
        f"✅ {n_t} tareas | "
        f"🔑 {n_d} decisiones"
    )

    return participation_df, tasks_df, decisions_df, status, md_path, json_path


def full_pipeline(
    meeting_text: str,
) -> tuple[str, pd.DataFrame, pd.DataFrame, pd.DataFrame, str, str | None, str | None]:
    """
    End-to-end: generate minutes with AI, then analyze the output.
    Returns (minutes_md, participation_df, tasks_df, decisions_df,
             status, md_path, json_path).
    """
    _empty = pd.DataFrame()

    if not meeting_text.strip():
        return "⚠️ Por favor pega el texto de la reunión.", _empty, _empty, _empty, "⚠️ Sin texto", None, None

    # Step 1 — Generate
    generated, gen_status = generate_minutes(meeting_text)
    if generated.startswith(("❌", "⚠️")):
        return generated, _empty, _empty, _empty, gen_status, None, None

    # Step 2 — Analyze the generated output
    participation_df, tasks_df, decisions_df, analysis_status, md_path, json_path = \
        analyze_minutes(generated)

    combined_status = f"🚀 {gen_status}  →  🔍 {analysis_status}"
    return generated, participation_df, tasks_df, decisions_df, combined_status, md_path, json_path

# ─────────────────────────────────────────────
# UI CONSTANTS
# ─────────────────────────────────────────────

_HEADER = """
# 📝 AI Meeting Minutes — Generator & Analyzer
**Genera y analiza actas de reunión automáticamente con IA**  
*Automatically generate and analyze meeting minutes with AI*
"""

_FOOTER = """
---
🤖 Modelo: `mistralai/Mistral-7B-Instruct-v0.2` &nbsp;|&nbsp;
🔍 Análisis: Regex (ES + EN) &nbsp;|&nbsp;
💡 Tip: la pestaña **Pipeline Completo** hace todo en un clic
"""

_GEN_PLACEHOLDER = (
    "Pega aquí la transcripción o notas de la reunión…\n"
    "Puede ser una conversación informal, notas rápidas o una transcripción.\n\n"
    "---\n\n"
    "Paste the meeting transcription or notes here…\n"
    "Can be an informal conversation, quick notes, or a full transcript."
)

_ANALYZE_PLACEHOLDER = (
    "Pega aquí el acta ya redactada para analizarla…\n"
    "Ejemplo: Tarea: revisar el informe. Responsable: Ana – 2024-03-15\n\n"
    "---\n\n"
    "Paste already-written meeting minutes here to analyze them.\n"
    "Example: Task: review the report. Responsible: Ana – 2024-03-15"
)

_WAITING = "Esperando entrada… / Waiting for input…"

# ─────────────────────────────────────────────
# GRADIO UI
# ─────────────────────────────────────────────

with gr.Blocks(
    title="AI Meeting Minutes Generator & Analyzer",
    theme=gr.themes.Soft(primary_hue="blue", secondary_hue="slate"),
) as app:

    gr.Markdown(_HEADER)

    with gr.Tabs():

        # ══════════════════════════════════════
        # TAB 1 — GENERATE
        # ══════════════════════════════════════
        with gr.TabItem("🚀 Generar Acta / Generate"):
            gr.Markdown(
                "### Usa IA para convertir transcripciones en actas estructuradas\n"
                "*⚠️ La generación puede tardar 4-5 minutos según la longitud del texto.*"
            )

            with gr.Row(equal_height=False):
                with gr.Column(scale=1):
                    gen_input = gr.Textbox(
                        label="🗣️ Texto de la reunión / Meeting Text",
                        lines=18,
                        placeholder=_GEN_PLACEHOLDER,
                    )
                    gen_btn = gr.Button(
                        "🚀 Generar Acta / Generate Minutes",
                        variant="primary",
                        size="lg",
                    )
                    gen_status = gr.Textbox(
                        label="📡 Estado / Status",
                        interactive=False,
                        value=_WAITING,
                    )

                with gr.Column(scale=1):
                    gen_output = gr.Markdown(
                        value=(
                            "*El acta generada aparecerá aquí…*\n"
                            "*Generated minutes will appear here…*"
                        )
                    )

            gen_btn.click(
                fn=generate_minutes,
                inputs=[gen_input],
                outputs=[gen_output, gen_status],
            )

        # ══════════════════════════════════════
        # TAB 2 — ANALYZE
        # ══════════════════════════════════════
        with gr.TabItem("🔍 Analizar Acta / Analyze"):
            gr.Markdown(
                "### Extrae tareas, decisiones y participación de un acta existente\n"
                "*Funciona con actas en español e inglés / Works with ES and EN minutes*"
            )

            with gr.Row():
                with gr.Column(scale=2):
                    analyze_input = gr.Textbox(
                        label="📋 Texto del acta / Minutes Text",
                        lines=12,
                        placeholder=_ANALYZE_PLACEHOLDER,
                    )
                with gr.Column(scale=1):
                    analyze_file = gr.File(
                        label="📁 O sube un archivo / Or upload a file (.txt / .md)",
                        file_types=[".txt", ".md"],
                        file_count="single",
                    )
                    gr.Markdown(
                        "_Si subes un archivo tiene prioridad sobre el texto pegado._\n"
                        "_Uploaded file takes priority over pasted text._"
                    )

            analyze_btn = gr.Button("🔍 Analizar / Analyze", variant="primary")
            analyze_status = gr.Textbox(
                label="📡 Estado / Status", interactive=False, value=_WAITING
            )

            with gr.Row():
                with gr.Column():
                    participation_table = gr.DataFrame(
                        label="👥 Participación / Participation", wrap=True
                    )
                with gr.Column():
                    decisions_table = gr.DataFrame(
                        label="🔑 Decisiones / Decisions", wrap=True
                    )

            tasks_table = gr.DataFrame(label="✅ Tareas / Tasks", wrap=True)

            gr.Markdown("### 💾 Exportar / Export")
            with gr.Row():
                export_md = gr.File(label="📄 Markdown (.md)")
                export_json = gr.File(label="📊 JSON (.json)")

            analyze_btn.click(
                fn=analyze_minutes,
                inputs=[analyze_input, analyze_file],
                outputs=[
                    participation_table,
                    tasks_table,
                    decisions_table,
                    analyze_status,
                    export_md,
                    export_json,
                ],
            )

        # ══════════════════════════════════════
        # TAB 3 — FULL PIPELINE
        # ══════════════════════════════════════
        with gr.TabItem("⚡ Pipeline Completo / Full Pipeline"):
            gr.Markdown(
                """
                ### Flujo completo en un clic / Full workflow in one click
                1. 🤖 La IA genera el acta desde la transcripción  
                2. 🔍 Se analiza automáticamente la salida  
                3. 💾 Descarga el resultado en MD o JSON  

                *⚠️ Puede tardar varios minutos / May take several minutes*
                """
            )

            pipeline_input = gr.Textbox(
                label="🗣️ Texto de la reunión / Meeting Text",
                lines=12,
                placeholder=_GEN_PLACEHOLDER,
            )

            pipeline_btn = gr.Button(
                "⚡ Ejecutar Pipeline Completo / Run Full Pipeline",
                variant="primary",
                size="lg",
            )
            pipeline_status = gr.Textbox(
                label="📡 Estado / Status", interactive=False, value=_WAITING
            )

            pipeline_minutes = gr.Markdown(
                value="*El resultado aparecerá aquí… / Results will appear here…*"
            )

            with gr.Row():
                with gr.Column():
                    pipeline_participation = gr.DataFrame(
                        label="👥 Participación / Participation", wrap=True
                    )
                with gr.Column():
                    pipeline_decisions = gr.DataFrame(
                        label="🔑 Decisiones / Decisions", wrap=True
                    )

            pipeline_tasks = gr.DataFrame(label="✅ Tareas / Tasks", wrap=True)

            gr.Markdown("### 💾 Exportar / Export")
            with gr.Row():
                pipeline_md = gr.File(label="📄 Markdown (.md)")
                pipeline_json = gr.File(label="📊 JSON (.json)")

            pipeline_btn.click(
                fn=full_pipeline,
                inputs=[pipeline_input],
                outputs=[
                    pipeline_minutes,
                    pipeline_participation,
                    pipeline_tasks,
                    pipeline_decisions,
                    pipeline_status,
                    pipeline_md,
                    pipeline_json,
                ],
            )

    gr.Markdown(_FOOTER)


if __name__ == "__main__":
    app.launch(server_name="0.0.0.0", share=False)
    st.download_button("Exportar HTML / HTML", generate_html(insights), file_name="meeting_insights.html")
    st.download_button("Exportar JSON / JSON", json.dumps(insights, indent=2, ensure_ascii=False), file_name="meeting_insights.json")

    st.success("✅ Análisis completado / Analysis completed")
