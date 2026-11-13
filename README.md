# 🔬 Research RAG Assistant

A robust, from-scratch Retrieval-Augmented Generation (RAG) system built specifically for academic researchers and students. This application allows users to upload PDF research papers and ask highly specific questions, receiving accurate, context-aware answers complete with direct citations to the source document's sections and deep reasoning traces.

Instead of relying on high-level wrappers like LangChain or LlamaIndex, the entire RAG pipeline is built **from scratch** to provide maximum control over parsing, semantic chunking, and context retrieval.

---

### 🌐 Live Application
👉 **[Research Assistance RAG System](https://rag-powered-research-assistance.streamlit.app/)**

---

## 🏆 Benchmark & Evaluation Results

The retrieval system was systematically evaluated across **45 specialized research papers** with **2,084 vector chunks** across 6 core vector search and ANN themes:

| Metric | Score | Context / Standard |
| :--- | :---: | :--- |
| **Top-1 Accuracy** | **83.33%** | 5 out of 6 domain research questions hit the exact ground-truth paper at Rank #1 |
| **Top-3 Accuracy** | **83.33%** | Consistent high precision at low K |
| **Top-5 Accuracy** | **100.00%** | 100% of queries found their target paper within the top 5 results |
| **Mean Reciprocal Rank (MRR)** | **0.8750** | Average rank placement of first relevant match is ~1.14 |
| **Average Search Latency** | **30.25 ms** | Real-time vector search across 2,084 768-d embeddings |
| **Disk Cache Ingestion Speed** | **< 1.00 ms** | SHA-256 disk caching eliminates redundant parsing |

---

## ✨ Features
- **Intelligent Academic Parsing:** Custom regex-based trimming removes boilerplate (like references and table of contents) and focuses strictly on the core content from Abstract/Introduction to Conclusion.
- **Context-Aware Semantic Chunking:** Text isn't blindly split. Chunks are generated section-by-section, ensuring no chunk bleeds across different paper headings.
- **Local Vector Database:** Uses ChromaDB for persistent local storage of document embeddings.
- **Deep Reasoning Traces:** Captures full `<think>...</think>` internal model reasoning steps in a collapsible inspection panel.
- **Source-Cited Answers:** The UI dynamically parses chunk IDs to provide clean citation chips, showing exactly which paper and section the LLM used.
- **Configurable Top-K Retrieval:** Interactive slider (3 to 30 chunks) to adjust context breadth.
- **Premium User Interface:** A sleek, dark-themed Streamlit UI featuring animated chat bubbles, transparent indexing status, and pre-built suggestion chips.

---

## 🛠️ Tech Stack
- **Frontend UI:** [Streamlit](https://streamlit.io/)
- **PDF Parsing:** `pymupdf4llm` (Extracts structural markdown directly from PDFs)
- **Embeddings:** `sentence-transformers` utilizing the `BAAI/bge-base-en-v1.5` model for state-of-the-art semantic search.
- **Vector Storage:** `chromadb` (Persistent Local Client)
- **Large Language Model (LLM):** OpenRouter API (`nvidia/nemotron-3-ultra-550b-a55b:free` or DeepSeek)

---

## 📐 Architecture

The system has two main flows: **Indexing** (processing uploaded PDFs into searchable vectors) and **Querying** (answering user questions from those vectors).

![RAG Architecture Diagram](rag_architecture.png)

### Deep Dive: The Custom RAG Pipeline
1. **Trimming (`paper_triming`):** Detects the start of "Abstract" or "Introduction" and the start of "Conclusion" or "References". Everything outside these bounds is discarded to prevent the LLM from hallucinating on citations.
2. **Structuring (`getting_content_between_headers`):** Groups text strictly under its recognized Markdown heading.
3. **Chunking (`chunks_generation`):** Creates overlapping 500-character chunks that are injected with their parent heading, guaranteeing that every chunk retains its structural context.
4. **Retrieval (`query_builder`):** Fetches the top-k most relevant chunks from ChromaDB and formats them into a strict prompt.
5. **Generation (`answer_generator`):** Sends the structured prompt to OpenRouter API with reasoning trace extraction.

---

## 🚀 Getting Started

### 1. Prerequisites
Make sure you have Python 3.10+ installed.

### 2. Installation
Clone the repository and install the required dependencies:
```bash
git clone https://github.com/Tatsat-Pathak/Research-Assistance-RAG-System.git
cd Research-Assistance-RAG-System
pip install -r requirements.txt
```

### 3. Environment Variables
Create a `.env` file in the root directory (or configure in Streamlit Cloud Secrets):
```env
OPENROUTER_API_KEY=your_openrouter_api_key_here
```

### 4. Run the Application
Launch the Streamlit server:
```bash
streamlit run app.py
```
Open the provided local URL (usually `http://localhost:8501`) in your browser, drag and drop research papers, click **⚡ Index Documents**, and ask questions!
