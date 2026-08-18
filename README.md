# LocalRAG

**Fully local RAG pipeline for PDF Q&A — no external LLM API required.**

Ask questions about your own PDF documents using open-source embedding
and language models. Everything runs locally in a Kaggle GPU environment,
with the complete pipeline from document processing to answer generation
and an interactive Gradio interface.

<p>
  <img alt="Python" src="https://img.shields.io/badge/python-3.10%2B-blue">
  <img alt="License" src="https://img.shields.io/badge/license-MIT-green">
  <img alt="GPU" src="https://img.shields.io/badge/GPU-T4%20(Kaggle)-76B900">
  <img alt="Status" src="https://img.shields.io/badge/status-active-brightgreen">
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

LocalRAG is a from-scratch Retrieval-Augmented Generation (RAG)
pipeline for asking natural-language questions about PDF documents.

The project combines semantic embeddings, vector similarity search, and
a locally running instruction-tuned language model to generate
context-grounded answers from uploaded documents.

Unlike API-based RAG applications, the language model runs locally in
the Kaggle GPU environment. No OpenAI, Anthropic, or other external LLM
API is required.

The project also includes an interactive Gradio interface that allows
users to upload a PDF, process it, ask questions, and view answers
along with the retrieved source pages.

| | |
|---|---|
| **Input** | One or more PDF files |
| **Output** | Grounded answers with cited source pages |
| **LLM** | Qwen2.5-1.5B-Instruct |
| **Embeddings** | all-MiniLM-L6-v2 |
| **Vector Search** | FAISS |
| **Interface** | Gradio |
| **Cost** | No external LLM API required |
| **Where it runs** | Kaggle Notebook with GPU |

---

## How it works

```text
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
   question ───► retrieve top-K chunks
                           │
                 ┌─────────▼──────────┐
                 │ Qwen2.5-1.5B-Instr.│
                 └─────────┬──────────┘
                           │
                    answer + [citations]
                           │
                 ┌─────────▼──────────┐
                 │    Gradio UI       │
                 └────────────────────┘
```

### Pipeline steps

1. **Extract & chunk** — `pypdf` extracts text from every page. Long
   pages are split into overlapping word chunks using a 900-word chunk
   size and 160-word overlap.

2. **Embed** — Each chunk is converted into a vector using
   `all-MiniLM-L6-v2` from `sentence-transformers`.

3. **Index** — The vectors are stored in a `faiss.IndexFlatIP` index
   for similarity-based retrieval.

4. **Retrieve** — A user's question is embedded using the same embedding
   model. FAISS retrieves the top-K most relevant document chunks.

5. **Generate** — `Qwen2.5-1.5B-Instruct` receives the retrieved chunks
   as context and generates an answer based on the available document
   context.

6. **Display** — Gradio provides an interactive interface for uploading
   PDFs, processing documents, asking questions, and viewing answers
   with source and page information.

---

## Tech stack

| Layer | Tool |
|---|---|
| Programming language | Python |
| PDF parsing | [`pypdf`](https://pypi.org/project/pypdf/) |
| Embedding model | [`all-MiniLM-L6-v2`](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2) |
| Embedding framework | [`sentence-transformers`](https://www.sbert.net/) |
| Vector search | [`faiss-cpu`](https://github.com/facebookresearch/faiss) |
| Language model | [`Qwen2.5-1.5B-Instruct`](https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct) |
| Model framework | [`transformers`](https://huggingface.co/docs/transformers/index) |
| Deep learning framework | [`PyTorch`](https://pytorch.org/) |
| User interface | [`Gradio`](https://www.gradio.app/) |
| Runtime | Kaggle Notebook with GPU |

---

## Quick start

### 1. Open the Kaggle Notebook

Open the project notebook in a Kaggle Notebook.

The notebook contains the complete RAG pipeline:

```text
PDF extraction
      ↓
Text chunking
      ↓
Embedding generation
      ↓
FAISS indexing
      ↓
Question retrieval
      ↓
Qwen generation
      ↓
Gradio interface
```

### 2. Enable GPU

In Kaggle:

```text
Settings → Accelerator → GPU
```

A GPU is required for efficient local inference with
`Qwen2.5-1.5B-Instruct`.

### 3. Enable Internet

Enable Internet in the Kaggle notebook settings.

Internet access is required initially to download the required packages
and model weights.

### 4. Add your PDF

Upload your PDF through the Kaggle environment or attach it as a Kaggle
Dataset.

The application will extract the text and prepare it for retrieval.

### 5. Run the notebook

Run all notebook cells in order.

The application will:

```text
PDF
 ↓
Extract text
 ↓
Create chunks
 ↓
Generate embeddings
 ↓
Build FAISS index
```

### 6. Launch the Gradio interface

After the required cells have been executed, the Gradio interface will
start.

The interface allows you to:

- Upload a PDF
- Process the document
- Ask questions about the document
- Receive an AI-generated answer
- View retrieved source pages

---

## Usage

### Gradio interface

The recommended way to use LocalRAG is through the Gradio interface.

#### Step 1 — Upload a PDF

Select a PDF document using the upload component.

#### Step 2 — Process the document

Click:

```text
Process Document
```

The application extracts the document text, creates chunks, generates
embeddings, and builds the FAISS vector index.

#### Step 3 — Ask a question

Enter a natural-language question about the uploaded document.

For example:

```text
What are the key findings in this document?
```

#### Step 4 — View the response

The system retrieves the most relevant chunks and provides them to Qwen
as context.

The response includes the generated answer and the retrieved source
information.

### Example

```text
User:
What are the key findings in this document?

Assistant:
The document describes the main findings related to ... [1][2]

Sources:
[1] report.pdf, page 3
[2] report.pdf, page 4
```

### Python usage

The core RAG pipeline can also be used directly from Python:

```python
answer_question(
    "What are the key findings in this document?"
)
```

This allows the retrieval and generation pipeline to be used without
the Gradio interface.

---

## Configuration

The main configuration values are defined near the beginning of the
notebook/application.

| Variable | Default | Description |
|---|---|---|
| `EMBEDDING_MODEL` | `all-MiniLM-L6-v2` | Sentence embedding model |
| `LLM_MODEL` | `Qwen2.5-1.5B-Instruct` | Local answer-generation model |
| `TOP_K` | `4` | Number of chunks retrieved per question |
| Chunk size | `900` words | Number of words in each chunk |
| Chunk overlap | `160` words | Overlap between consecutive chunks |

### Embedding model

The project uses:

```text
sentence-transformers/all-MiniLM-L6-v2
```

to convert document chunks and user questions into semantic vector
representations.

### Vector search

FAISS uses:

```python
faiss.IndexFlatIP(...)
```

for similarity-based retrieval.

The embeddings are normalized before indexing so inner-product
similarity can be used for cosine-style similarity search.

### Language model

The project uses:

```text
Qwen/Qwen2.5-1.5B-Instruct
```

for local answer generation.

The model receives the retrieved document context together with the
user's question.

---

## Project structure

```text
local-rag/
│
├── document_qa_rag.ipynb    # Complete RAG notebook
├── app.py                   # Application version of the RAG pipeline
├── requirements.txt         # Python dependencies
├── README.md                # Project documentation
└── .gitignore               # Ignored files
```

### `document_qa_rag.ipynb`

Contains the complete RAG implementation, including:

```text
PDF extraction
      ↓
Text preprocessing
      ↓
Chunk creation
      ↓
Embedding generation
      ↓
FAISS vector index
      ↓
Semantic retrieval
      ↓
Qwen answer generation
      ↓
Gradio interface
```

### `app.py`

Contains the application-oriented version of the RAG pipeline for
running the project outside the development notebook.

### `requirements.txt`

Contains the Python packages required to run the application.

### `README.md`

Contains project documentation, setup instructions, architecture,
usage information, and troubleshooting details.

### `.gitignore`

Contains files and directories that should not be committed to Git.

---

## Troubleshooting

| Problem | Fix |
|---|---|
| `AssertionError: Enable GPU` | Enable a GPU accelerator in Kaggle settings |
| Out of GPU memory | Use a smaller compatible Qwen model or reduce the retrieved context |
| `No PDFs found` | Check that the PDF is correctly uploaded or attached |
| Blank extracted text | The PDF may be scanned/image-based and may require OCR |
| Model download fails | Confirm that Internet access is enabled |
| Gradio does not start | Check the installed Gradio version and required dependencies |
| PDF processing fails | Verify that the PDF contains extractable text |
| Poor answers | Check the retrieved chunks and similarity scores |
| Irrelevant retrieval | Adjust chunk size, overlap, or `TOP_K` |
| Slow inference | Ensure that the Kaggle GPU accelerator is enabled |

### GPU memory issues

If Qwen does not fit into the available GPU memory, use a smaller
compatible language model or reduce the amount of retrieved context.

For example:

```text
Qwen2.5-0.5B-Instruct
```

The rest of the RAG pipeline can remain unchanged.

### Scanned PDFs

The current PDF extraction pipeline relies on text extraction.

If a document contains scanned images instead of selectable text, OCR
preprocessing may be required before using it with LocalRAG.

---

## Roadmap

- [ ] Improve chunking strategies for different document types
- [ ] Add support for multiple documents
- [ ] Add document-level filtering during retrieval
- [ ] Improve retrieval quality with reranking
- [ ] Add OCR support for scanned PDFs
- [ ] Add conversation memory
- [ ] Add evaluation metrics for retrieval and answer quality
- [ ] Experiment with different embedding models
- [ ] Experiment with different local instruction-tuned LLMs
- [ ] Improve the Gradio interface
- [ ] Add persistent vector-store support
- [ ] Deploy the application as a publicly accessible web application

---

## License

Released under the [MIT License](LICENSE).

---

**LocalRAG** — A fully local Retrieval-Augmented Generation pipeline
for intelligent PDF question answering.
