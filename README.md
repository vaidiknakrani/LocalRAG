# LocalRAG

**Fully local RAG pipeline for PDF Q&A — no API keys, no cloud costs.**

Ask questions about your own PDF documents using open-source embedding
and language models. Everything runs locally on a free Kaggle GPU — your
documents and questions never leave the session.

<p>
  <img alt="Python" src="https://img.shields.io/badge/python-3.10%2B-blue">
  <img alt="License" src="https://img.shields.io/badge/license-MIT-green">
  <img alt="GPU" src="https://img.shields.io/badge/GPU-T4%20(Kaggle)-76B900">
  <img alt="Status" src="https://img.shields.io/badge/status-educational-yellow">
</p>

---

## Table of contents

- [Overview](#overview)
- [How it works](#how-it-works)
- [Tech stack](#tech-stack)
- [Quick start](#quick-start)
- [Usage](#usage)
- [Configuration](#configuration)
- [Project structure](#project-structure)
- [Troubleshooting](#troubleshooting)
- [Roadmap](#roadmap)
- [License](#license)

---

## Overview

LocalRAG is a minimal, from-scratch Retrieval-Augmented Generation (RAG)
pipeline for asking natural-language questions about PDF documents. Unlike
most RAG tutorials, it uses **no paid API** — every model runs locally on
a free Kaggle GPU session, making it reproducible at zero cost.

| | |
|---|---|
| **Input** | One or more PDF files |
| **Output** | Grounded answers with cited source pages |
| **Cost** | $0 — no OpenAI/Anthropic/HF API calls |
| **Where it runs** | Kaggle Notebook (GPU T4) |

## How it works

```
                 ┌────────────────────┐
   PDF(s)  ───►  │  extract & chunk   │
                 └─────────┬──────────┘
                            │
                 ┌─────────▼──────────┐
                 │  embed (MiniLM)    │
                 └─────────┬──────────┘
                            │
                 ┌─────────▼──────────┐
                 │  FAISS vector index│
                 └─────────┬──────────┘
                            │
   question ───►  retrieve top-K chunks
                            │
                 ┌─────────▼──────────┐
                 │ Qwen2.5-1.5B-Instr.│
                 └─────────┬──────────┘
                            │
                    answer + [citations]
```

1. **Extract & chunk** — `pypdf` pulls text from every page; long pages are
   split into overlapping word chunks (900 words, 160-word overlap) so
   context isn't lost at chunk boundaries.
2. **Embed** — each chunk is converted into a vector with
   `all-MiniLM-L6-v2` (`sentence-transformers`).
3. **Index** — vectors are stored in a `faiss.IndexFlatIP` index for fast
   cosine-similarity search.
4. **Retrieve** — a question is embedded the same way and matched against
   the index to pull the top-K most relevant chunks.
5. **Generate** — `Qwen2.5-1.5B-Instruct` receives the retrieved chunks as
   context and is instructed to answer *only* from that context, citing
   sources like `[1]`, `[2]`.

## Tech stack

| Layer | Tool |
|---|---|
| PDF parsing | [`pypdf`](https://pypi.org/project/pypdf/) |
| Embedding model | [`all-MiniLM-L6-v2`](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2) |
| Vector search | [`faiss-cpu`](https://github.com/facebookresearch/faiss) |
| Language model | [`Qwen2.5-1.5B-Instruct`](https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct) |
| Runtime | Kaggle Notebook, GPU T4 |

## Quick start

1. Open a new **Kaggle Notebook**.
2. **Settings → Accelerator → GPU T4 x2**
3. **Settings → Internet → On** *(first run only — downloads model weights)*
4. Upload your PDF(s) as a **Kaggle Dataset**, then attach it via **Add Input**.
5. Point the notebook at your dataset:
   ```python
   DOCUMENT_FOLDER = Path('/kaggle/input/<your-dataset-folder>')
   ```
6. **Run all cells** in order.

## Usage

```python
answer_question("What are the key findings in this document?")
```

**Example output:**

```
Answer:
 The document describes ... [1][2]

Retrieved passages:
[1] report.pdf, page 3 (similarity 0.812)
[2] report.pdf, page 4 (similarity 0.774)
```

Run the cell again with a new question any time — the model and index
stay in memory, so repeat queries are fast.

## Configuration

All settings live in one config cell near the top of the notebook:

| Variable | Default | Description |
|---|---|---|
| `EMBEDDING_MODEL` | `all-MiniLM-L6-v2` | Sentence embedding model |
| `LLM_MODEL` | `Qwen2.5-1.5B-Instruct` | Local answer-generation model |
| `TOP_K` | `4` | Number of chunks retrieved per question |
| chunk size / overlap | `900` / `160` words | Controls context granularity |

## Project structure

```
local-rag/
├── notebook.ipynb     # main pipeline (extract → embed → index → answer)
├── README.md
└── LICENSE
```

## Troubleshooting

| Problem | Fix |
|---|---|
| `AssertionError: Enable GPU` | Settings → Accelerator → GPU T4 |
| Out of GPU memory | Switch `LLM_MODEL` to `Qwen2.5-0.5B-Instruct`, or lower `TOP_K` |
| `No PDFs found` | Check `DOCUMENT_FOLDER` matches your attached dataset's exact folder name |
| Blank extracted text | PDF is likely scanned/image-based — OCR it first (e.g. `pytesseract`) before use |
| Model download fails | Confirm Internet is enabled in notebook settings |

## Roadmap

- [ ] Swap `IndexFlatIP` for `IndexIVFFlat` to scale beyond small document sets
- [ ] Add per-document chunking strategy (currently uniform word-count based)
- [ ] Add a simple Gradio/Streamlit UI for interactive Q&A
- [ ] Support offline model loading via a private Kaggle model dataset

## License

Released under the [MIT License](LICENSE).

---

*Educational project — built as part of an AI/ML bootcamp exercise on
Retrieval-Augmented Generation. Not intended for production-scale
document retrieval.*
