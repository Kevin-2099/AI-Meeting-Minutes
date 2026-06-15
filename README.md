# 📝 AI Meeting Minutes Generator & Analyzer

Genera y analiza actas de reuniones automáticamente utilizando Inteligencia Artificial y extracción de información basada en reglas.

Este proyecto combina:

* 🤖 Generación automática de actas mediante Mistral-7B-Instruct.
* 🔍 Extracción de participantes, tareas, responsables, fechas y decisiones.
* ⚡ Pipeline completo que transforma transcripciones en información accionable.
* 💾 Exportación de resultados en Markdown y JSON.

Compatible con reuniones en español e inglés.

---

## 🚀 Características

### 🤖 Generación de actas con IA

Convierte transcripciones o notas de reuniones en actas estructuradas en formato Markdown.

**Secciones generadas automáticamente:**

* Participantes
* Temas tratados
* Decisiones tomadas
* Acciones y responsables
* Próximos pasos

**Características:**

* Detección automática de idioma.
* Salidas estructuradas y consistentes.
* Generación determinista.
* No inventa información cuando no existe en el texto.

---

### 🔍 Análisis automático de actas

Analiza actas ya redactadas y extrae automáticamente:

* 👥 Participantes
* ✅ Tareas
* 👤 Responsables
* 📅 Fechas límite
* 🔑 Decisiones

Compatible con documentos en:

* Español 🇪🇸
* Inglés 🇬🇧

---

### ⚡ Pipeline Completo

Permite ejecutar todo el flujo en un solo paso:

```text
Transcripción
      ↓
Generación con IA
      ↓
Acta estructurada
      ↓
Análisis automático
      ↓
Participantes + Tareas + Decisiones
      ↓
Exportación
```

---

## 🏗️ Arquitectura

```text
                 ┌─────────────────┐
                 │ Transcripción   │
                 │ de reunión      │
                 └────────┬────────┘
                          │
                          ▼
                ┌──────────────────┐
                │ Mistral-7B       │
                │ Generador Acta   │
                └────────┬─────────┘
                         │
                         ▼
                ┌──────────────────┐
                │ Acta Markdown    │
                └────────┬─────────┘
                         │
                         ▼
                ┌──────────────────┐
                │ Analizador Regex │
                └────────┬─────────┘
                         │
      ┌──────────────────┼──────────────────┐
      ▼                  ▼                  ▼
Participantes       Tareas            Decisiones
```

---

## 🧠 Modelo utilizado

**Modelo:**

```text
mistralai/Mistral-7B-Instruct-v0.2
```

**Inferencia mediante:**

```python
transformers.pipeline(
    "text-generation",
    model="mistralai/Mistral-7B-Instruct-v0.2"
)
```

**Configuración:**

```python
temperature = 0
do_sample = False
```

Esto permite obtener resultados consistentes y reproducibles.

---

## 💻 Instalación

### 1️⃣ Clonar el repositorio

```bash
git clone https://github.com/tu-usuario/ai-meeting-minutes.git
cd ai-meeting-minutes
```

### 2️⃣ Crear entorno virtual

#### Linux / macOS

```bash
python -m venv .venv
source .venv/bin/activate
```

#### Windows

```bash
python -m venv .venv
.venv\Scripts\activate
```

### 3️⃣ Instalar dependencias

```bash
pip install -r requirements.txt
```

---

## ▶️ Ejecución local

```bash
python app.py
```

La aplicación estará disponible en:

```text
http://localhost:7860
```

---

## 📥 Entradas soportadas

### Generación

* Transcripciones de reuniones
* Notas de reuniones
* Conversaciones informales

### Análisis

* Texto pegado manualmente
* Archivos `.txt`
* Archivos `.md`

---

## 📤 Exportación

Los resultados pueden descargarse en:

| Formato           | Descripción             |
| ----------------- | ----------------------- |
| 📄 Markdown (.md) | Informe legible         |
| 📊 JSON (.json)   | Procesamiento posterior |

---

## 🌍 Idiomas soportados

* Español 🇪🇸
* Inglés 🇬🇧

La detección de idioma se realiza automáticamente mediante:

```text
langdetect
```

---

## 🧩 Casos de uso

* 📈 Reuniones de trabajo
* 📋 Gestión de proyectos
* 🔄 Metodologías ágiles
* 🏢 Comités
* 🎓 Clases y reuniones académicas
* 🎤 Entrevistas
* 🧠 Workshops
* 📚 Documentación interna

---

## ⚠️ Limitaciones

* La calidad depende de la calidad de la transcripción.
* Algunas expresiones muy informales pueden no ser detectadas por las reglas regex.
* Reuniones extensas pueden requerir varios minutos de procesamiento.
* El modelo requiere recursos suficientes para ejecutarse localmente.

---

## 🛠️ Tecnologías utilizadas

* Python
* Gradio
* Transformers
* Mistral-7B-Instruct
* Pandas
* LangDetect
* Regex (Expresiones regulares)

---

## 📄 Licencia
Este proyecto se distribuye bajo una licencia propietaria con acceso al código (source-available).

El código fuente se pone a disposición únicamente para fines de visualización, evaluación y aprendizaje.

❌ No está permitido copiar, modificar, redistribuir, sublicenciar, ni crear obras derivadas del software o de su código fuente sin autorización escrita expresa del titular de los derechos.

❌ El uso comercial del software, incluyendo su oferta como servicio (SaaS), su integración en productos comerciales o su uso en entornos de producción, requiere un acuerdo de licencia comercial independiente.

📌 El texto legalmente vinculante de la licencia es la versión en inglés incluida en el archivo LICENSE.

Se proporciona una traducción al español en LICENSE_ES.md únicamente con fines informativos. En caso de discrepancia, prevalece la versión en inglés.

---

## Autor
Kevin-2099
