# 🔬 AI-Powered Research Assistance RAG System

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://rag-powered-research-assistance.streamlit.app/)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![ChromaDB](https://img.shields.io/badge/vector--db-ChromaDB-orange.svg)](https://www.trychroma.com/)
[![Embeddings](https://img.shields.io/badge/embeddings-BAAI%2Fbge--base--en--v1.5-purple.svg)](https://huggingface.co/BAAI/bge-base-en-v1.5)
[![LLM](https://img.shields.io/badge/LLM-Nemotron%203.5-green.svg)](https://openrouter.ai/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A robust, from-scratch **Retrieval-Augmented Generation (RAG)** assistant engineered specifically for academic researchers and students. This application allows users to upload PDF research papers and ask nuanced, high-level or fine-grained questions, receiving synthesized, context-aware answers complete with **direct section citations** and **deep reasoning traces**.

Instead of relying on rigid high-level abstractions like LangChain or LlamaIndex, the entire RAG pipeline is built **from scratch** to give full programmatic control over layout parsing, academic boundary trimming, structure-aware semantic chunking, and vector retrieval.

---

### 🌐 Live Application
> ⚠️ The Streamlit Cloud deployment linked below is currently returning "not found" for visitors (checked 2026-11-14). It needs to be redeployed — see the hosting notes in the repo history. Run it locally in the meantime (see Getting Started).

👉 **Access the Live Web Assistant:** **[rag-powered-research-assistance.streamlit.app](https://rag-powered-research-assistance.streamlit.app/)**

---

## 🏆 Benchmark & Retrieval Evaluation

The RAG retrieval engine was systematically evaluated and benchmarked across **45 specialized research papers** comprising **2,084 vector chunks** across 6 core vector search themes:
1. **Foundations Papers** (HNSW, NSW, Product Quantization, FAISS, CAGRA)
2. **GPU Papers** (CAGRA, GGNN, SONG, GPU-accelerated OPQ)
3. **Core Papers** (Variance-Aware Quantization, Search Quality Trade-offs, AQR-HNSW)
4. **Auto-Tuning Papers** (Constrained Optimization, HNSW Hyperparameter Tuning)
5. **Resource Constraint** (DiskANN Single-Node, Raspberry Pi & Edge ANN)
6. **Server-Grade Benchmarking** (High-Dimensional ANN, KScaNN, Unified Vector DB Benchmarks)

### Verified Retrieval Performance Metrics

| Metric | Score | Context / Standard |
| :--- | :---: | :--- |
| **Top-1 Retrieval Accuracy** | **83.33%** | 5 out of 6 domain research questions hit the exact ground-truth paper at Rank #1 |
| **Top-3 Retrieval Accuracy** | **83.33%** | High precision maintained at low context thresholds |
| **Top-5 Retrieval Accuracy** | **100.00%** | 100% of domain queries found their target paper within the top 5 results |
| **Mean Reciprocal Rank (MRR)** | **0.8750** | Average rank placement of first relevant match is ~1.14 |
| **Average Search Latency** | **30.25 ms** | Real-time cosine similarity search across 2,084 768-d embeddings |
| **Disk Cache Ingestion Speed** | **< 1.00 ms** | SHA-256 disk hashing eliminates redundant parsing |

> Note: these figures are from the author's local evaluation. The evaluation script and data are not in this repo, so the numbers can't be reproduced from here alone.

---

## 📐 System Architecture

The pipeline consists of two synchronized, decoupled workflows: **Indexing Flow** (document processing & persistent vector storage) and **Querying Flow** (context retrieval, prompt grounding & LLM reasoning).

![RAG System Architecture](rag_architecture.png)

---

### Deep Dive: The End-to-End Pipeline

#### 1. Ingestion & Indexing Flow
1. **Upload PDF Files:** Multi-file drag-and-drop ingestion via the Streamlit interface with persistent disk caching.
2. **Layout Parsing & OCR:** Converts complex PDF documents into structured Markdown while preserving tables and section hierarchies via MinerU layout analysis.
3. **Academic Trimming (`paper_triming`):** Intelligent regex heuristics locate the start of "Abstract" or "Introduction" and the start of "Conclusion" or "References". Non-core boilerplate (cover pages, headers, footers, references, and appendices) is stripped to prevent the LLM from hallucinating on citation lists.
4. **Extract Sections & Tables (`getting_content_between_headers`):** Structures text strictly under its parent Markdown section heading and isolates tabular markdown data.
5. **Structure-Aware Chunking (`chunks_generation`):** Generates overlapping ~300-word chunks with sentence boundary preservation. Each chunk is tagged with its parent section heading, preventing chunks from bleeding across different headers. The size is set in `app.py` (`chunk_size=300`, in words) and is passed through to `chunks_generated.py`.
6. **Dense Vector Embeddings (`create_embedding_text`):** Transforms chunks into normalized 768-dimensional dense vectors using the state-of-the-art `BAAI/bge-base-en-v1.5` model.
7. **ChromaDB Vector Store (`initialize_vector_database`):** Chunks, embeddings, and document IDs are stored persistently in a local ChromaDB collection using cosine similarity scoring.

#### 2. Retrieval & Querying Flow
1. **User Research Question:** The researcher enters a query in the chat interface or selects a suggested research prompt.
2. **Embed Query Vector:** The user's query is normalized and embedded into a 768-d vector using `BAAI/bge-base-en-v1.5`.
3. **ChromaDB Vector Search:** Executes an approximate nearest neighbor (ANN) search against indexed vectors.
4. **Top-K Chunks Retrieved:** Retrieves the most relevant chunks based on the user-configurable Top-K slider (3 to 30 chunks).
5. **Augmented Prompt Construction (`query_builder`):** Formulates a grounded context window pairing retrieved chunks with document titles, section headers, and strict grounding instructions.
6. **LLM Deep Reasoning (`answer_generator`):** Dispatches the prompt to OpenRouter API (default model `nvidia/nemotron-3.5-super-vl:free`, overridable with the `RAG_LLM_MODEL` env var), capturing `<think>...</think>` internal reasoning traces.
7. **Cited Synthesis & Reasoning Output:** Renders the synthesized response in the dark-themed UI, surfaces clickable source chips (Paper Title + Section Heading), and exposes the reasoning steps in an expandable drawer.

---

## ✨ Key Features

- **Deep Reasoning Traces:** Full inspection of internal model reasoning steps (`<think>...</think>`) in a collapsible drawer.
- **Granular Source Citations:** Every response surfaces interactive chips showing the exact document title, section heading, and whether the evidence originated from a table.
- **Configurable Top-K Slider:** Adjust context breadth from 3 to 30 chunks directly from the sidebar.
- **Structure-Aware Semantic Chunking:** Chunks respect paragraph and section boundaries, preventing semantic fragmentation.
- **Instant Ingestion with Caching:** SHA-256 disk cache prevents redundant parsing on repeated files (<1ms reload).
- **Executive Dark UI:** Custom glassmorphism CSS theme built with Inter typography, animated response bubbles, and clean status indicators.

---

## 🛠️ Tech Stack

| Component | Technology | Role |
| :--- | :--- | :--- |
| **Frontend UI** | [Streamlit](https://streamlit.io/) | Reactive web application with custom CSS styling |
| **Document Parsing** | `pymupdf4llm` / MinerU | Layout-aware structural markdown & table extraction |
| **Embedding Model** | `BAAI/bge-base-en-v1.5` | 768-dimensional normalized dense semantic representations |
| **Vector Database** | [ChromaDB](https://www.trychroma.com/) | Persistent local vector client with cosine distance indexing |
| **LLM Backend** | OpenRouter API | Reasoning model (`nvidia/nemotron-3.5-super-vl:free` by default; set `RAG_LLM_MODEL` to change it) |
| **Environment** | Python 3.10+ | Core execution runtime |

---

## 🚀 Getting Started

### 1. Prerequisites
Ensure you have **Python 3.10** or higher installed.

### 2. Clone & Install
```bash
# Clone the repository
git clone https://github.com/Tatsat-Pathak/Research-Assistance-RAG-System.git
cd Research-Assistance-RAG-System

# Install dependencies
pip install -r requirements.txt
```

### 3. Configure API Key
Create a `.env` file in the root directory:
```env
OPENROUTER_API_KEY=your_openrouter_api_key_here
```
Or set it as a secret on Streamlit Cloud (`OPENROUTER_API_KEY` in Secrets). There is deliberately no key input in the app UI — the key must never be typed into, or displayed by, the interface. Without a key, uploading and indexing still work, but asking questions will show a clear "no API key set" error instead of a cryptic API failure.

### 4. Run Locally
```bash
streamlit run app.py
```
Open your browser at `http://localhost:8501`, drop your PDF papers into the sidebar, click **⚡ Index Documents**, and ask your research questions!

---

## 📂 Repository Structure

```
Research-Assistance-RAG-System/
├── app.py                     # Streamlit frontend with custom dark theme & reasoning UI
├── RAG_Pipeline.py            # Core RAG logic: parsing, chunking, embedding & OpenRouter LLM
├── extract_papers.py          # Section/table/equation extraction from MinerU structured output
├── chunks_generated.py        # Structure-aware chunking (~300-word chunks, section paths)
├── rag_architecture.png       # High-resolution system architecture diagram
├── requirements.txt           # Pinned/bounded production dependencies
├── LICENSE                    # MIT License
├── .gitignore                 # Excludes caches, vector databases, and private PDFs
└── README.md                  # Project documentation & benchmark evaluation results
```

### Hosting notes
- PDFs are parsed **sequentially by default** (one document at a time). Each parallel MinerU worker loads its own copy of the parser models, so parallel parsing can use several times the RAM and crash a small host. Set `RAG_MAX_WORKERS=2` only on a host with RAM to spare.
- Uploaded PDFs, the parse cache, and the ChromaDB index live in local folders (`uploaded_pdfs/`, `parse_cache/`, `upload_pdfs_chroma_db/`) and are **not persistent** on hosted platforms that wipe disk on restart/sleep — after a cold start, papers must be uploaded and indexed again.

---

## 📄 License
Distributed under the **MIT License**. See `LICENSE` for more information.
