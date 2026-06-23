# RAG Learning: From Basics to Production-Ready Advanced Pipelines

> **A complete, hands-on journey through Retrieval-Augmented Generation — from foundational ingestion to advanced multi-modal, multi-query, hybrid retrieval systems with RRF fusion and reranking.**

[![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)](https://www.python.org/)
[![LangChain](https://img.shields.io/badge/LangChain-0.2+-green.svg)](https://www.langchain.com/)
[![ChromaDB](https://img.shields.io/badge/VectorDB-Chroma-purple.svg)](https://www.trychroma.com/)
[![Groq](https://img.shields.io/badge/LLM-Groq-orange.svg)](https://groq.com/)

---

## Table of Contents

- [Overview](#overview)
- [What You'll Learn](#what-youll-learn)
- [Architecture & Pipeline](#architecture--pipeline)
  - [Phase 1: Ingestion Pipeline](#phase-1-ingestion-pipeline)
  - [Phase 2: Chunking Strategies (5+ Types)](#phase-2-chunking-strategies)
  - [Phase 3: Advanced Retrieval Methods](#phase-3-advanced-retrieval-methods)
  - [Phase 4: Multi-Query RAG with RRF](#phase-4-multi-query-rag-with-rrf)
  - [Phase 5: Hybrid Search & Reranking](#phase-5-hybrid-search--reranking)
  - [Phase 6: Multi-Modal RAG](#phase-6-multi-modal-rag)
- [Project Structure](#project-structure)
- [Installation & Setup](#installation--setup)
- [Usage Guide](#usage-guide)
- [Course Attribution](#course-attribution)
- [Key Concepts Explained](#key-concepts-explained)
- [Results & Outputs](#results--outputs)

---

## Overview

This repository documents my complete learning journey through **Retrieval-Augmented Generation (RAG)** — from understanding the basic building blocks to implementing **production-grade advanced pipelines** that handle complex documents, multiple query variations, hybrid retrieval, and intelligent reranking.

Every script and notebook in this repo is **runnable, well-commented, and designed for learning**. The codebase progresses logically from simple to advanced, making it perfect for anyone who wants to **master RAG from first principles to enterprise patterns**.

---

## What You'll Learn

| Level | Topics Covered |
|-------|---------------|
| **Beginner** | Document loading, basic chunking, vector stores, similarity search, simple RAG chains |
| **Intermediate** | Semantic chunking, agentic chunking, MMR retrieval, history-aware generation, score thresholds |
| **Advanced** | Multi-query generation, Reciprocal Rank Fusion (RRF), Hybrid Search (Dense + Sparse), Cohere Reranking, Multi-modal RAG (PDFs with tables, images, and text) |

---

## Architecture & Pipeline

### Phase 1: Ingestion Pipeline (`src/01-ingestion_pipeline.py`)

The foundation of any RAG system. This pipeline demonstrates:

```
Data Directory (./data/)
    |-- Google.txt
    |-- Microsoft.txt
    |-- Nvidia.txt
    |-- SpaceX.txt
    |-- Tesla.txt
         |
    +-------------------------------------+
    |  1. LOAD: DirectoryLoader + TextLoader |
    |     (UTF-8 encoding, glob patterns)   |
    |         |                             |
    |  2. CHUNK: CharacterTextSplitter      |
    |     (chunk_size=1000, overlap=0)      |
    |         |                             |
    |  3. EMBED: HuggingFaceEmbeddings        |
    |     (all-MiniLM-L6-v2, batch=32)      |
    |         |                             |
    |  4. STORE: Chroma Vector DB           |
    |     (cosine similarity, HNSW index)   |
    |     Persisted to: db/chroma_db          |
    +-------------------------------------+
```

**Key Features:**
- Uses **HuggingFace embeddings** (free, local, privacy-preserving)
- **Cosine similarity** space for semantic matching
- **Persistent storage** with collection metadata
- Environment-based API key management via `.env`

---

### Phase 2: Chunking Strategies (5+ Types)

This repository implements **every major chunking strategy** used in production RAG:

#### 2.1 Character Text Splitter (`src/05-character_text_splitter.py`)
- Simple fixed-size splitting by character count
- Demonstrates the **limitations** of naive splitting (cuts mid-sentence)

#### 2.2 Recursive Character Text Splitter (`src/05-character_text_splitter.py`)
- Hierarchical separator-based splitting: `["\n\n", "\n", ". ", " ", ""]`
- Respects natural text boundaries
- **The most commonly used production chunker**

#### 2.3 Semantic Chunking (`src/06-semantic_chunking.py`)
```python
SemanticChunker(
    embeddings=embedding_model,
    breakpoint_threshold_type="percentile",  # Detects semantic shifts
    breakpoint_threshold_amount=70
)
```
- Groups sentences by **meaning similarity** using embeddings
- Automatically detects topic transitions
- Produces **coherent, contextually unified chunks**

#### 2.4 Agentic Chunking (`src/07-agentic_chunking.py`)
- Uses **LLM intelligence** (Llama 3.3 70B via Groq) to split text
- LLM places `<<<SPLIT>>>` markers at **logical topic boundaries**
- Most flexible — understands document structure semantically

#### 2.5 Multi-Modal Chunking (`src/08-multi_model_rag.ipynb`)
- **PDF ingestion** with mixed content: text, tables, images
- Uses **Unstructured library** to extract atomic elements
- **Title-based chunking** for document structure preservation
- **Content-type separation**: detects images, tables, raw text
- **LLM-generated summaries** for each content type
- Stores raw data in metadata, summaries in `page_content`

```
PDF (tables + images + text)
    |
Unstructured Library Extraction
    |
Chunk by Document Title
    |
+-----------------------------------------+
|  Content Type Detection per Chunk       |
|  |-- Images -> LLM Visual Summary       |
|  |-- Tables -> LLM Structured Summary   |
|  |-- Text   -> LLM Textual Summary     |
|         |                               |
|  LangChain Documents Created            |
|  * page_content = summary               |
|  * metadata = {raw_image, raw_table,   |
|                source, type, ...}       |
+-----------------------------------------+
```

---

### Phase 3: Advanced Retrieval Methods (`src/09-retrieval_methods.py`)

Four distinct retrieval strategies demonstrated:

| Method | Code | Use Case |
|--------|------|----------|
| **Similarity Search** | `search_kwargs={"k": 3}` | Basic semantic matching |
| **Score Threshold** | `similarity_score_threshold` + `score_threshold=0.3` | Quality-gated retrieval (filter out weak matches) |
| **MMR (Max Marginal Relevance)** | `search_type="mmr"`, `fetch_k=10`, `lambda_mult=0.5` | Balance relevance vs. diversity |
| **History-Aware** | (`src/03-history_aware_generation.py`) | Rewrite queries based on chat history for context continuity |

**MMR Formula:** `Score = lambda * Relevance - (1-lambda) * Redundancy`
- `lambda_mult=0.5`: Equal weight to relevance and diversity
- `fetch_k=10`: Consider 10 candidates, return top `k=3`

**History-Aware Pattern:**
```
User Query + Chat History -> LLM Rewrites -> Standalone Query -> Retrieve -> Generate
```

---

### Phase 4: Multi-Query RAG with RRF (`src/10-multi_query_retrieval.py`, `src/11-reciprocal_rank_fusion.py`)

This is where the pipeline becomes **truly advanced**:

#### Step 1: Query Variation Generation (LLM + Pydantic + JSON Mode)
```python
class QueryVariations(BaseModel):
    queries: List[str]

# Groq native JSON mode ensures structured output
response = llm.chat.completions.create(
    model="llama-3.3-70b-versatile",
    response_format={"type": "json_object"}
)
# Generates 3 semantically equivalent but syntactically different queries
```

#### Step 2: Parallel Retrieval
- Each query variation retrieves `k=5` documents independently
- Captures **different lexical and semantic spaces**

#### Step 3: Reciprocal Rank Fusion (RRF) (`src/11-reciprocal_rank_fusion.py`)

```python
def reciprocal_rank_fusion(chunk_lists, k=60):
    # RRF Score = sum of 1/(k + rank_position) for each query
    # Documents appearing in MULTIPLE query results get BOOSTED
    # Higher positions in ANY list contribute more
    # k=60 dampens the impact of absolute ranking differences
```

**Why RRF is Powerful:**
- **No training required** — purely algorithmic
- **Handles different retrieval scores** across queries (normalizes ranks)
- **Boosts consensus** — documents found by multiple queries rank higher
- **Deduplicates automatically** — same chunk from different queries merges scores

**Example Output:**
```
RANK 1 (RRF Score: 0.0452) — Found in Query 1 (pos 2) + Query 3 (pos 3)
RANK 2 (RRF Score: 0.0381) — Found in Query 2 (pos 1) only
```

---

### Phase 5: Hybrid Search & Reranking (`src/12-hybrid_search.ipynb`, `src/13-reranker.ipynb`)

#### 5.1 Hybrid Search: Dense + Sparse Retrieval

```python
from langchain.retrievers import EnsembleRetriever
from langchain_community.retrievers import BM25Retriever

hybrid_retriever = EnsembleRetriever(
    retrievers=[vector_retriever, bm25_retriever],
    weights=[0.5, 0.5]  # Equal weight to semantic and keyword
)
```

| Retriever | Type | Strength |
|-----------|------|----------|
| **Vector (Chroma)** | Dense/Semantic | Understands meaning, synonyms, concepts |
| **BM25** | Sparse/Keyword | Exact matches, rare terms, proper nouns |

**The Hybrid Advantage:**
- "SpaceX" -> BM25 finds exact mentions; Vector finds "space exploration company"
- "Java" (programming vs. coffee) -> Combined disambiguation
- "Apple" (company vs. fruit) -> Contextual ranking

#### 5.2 Reranking with Cohere (`src/13-reranker.ipynb`)

**Two-Stage Architecture:**
```
Stage 1 (Fast): Hybrid Retrieval -> Top 30-100 candidates
                      |
Stage 2 (Accurate): Cohere Reranker -> Re-scores and reorders
                      |
              Top K most relevant to LLM
```

```python
from langchain_cohere import CohereRerank

reranker = CohereRerank(
    model="rerank-english-v3.0",
    top_n=5
)
```

**Why Rerankers Matter:**
- Vector similarity is **coarse-grained** (embeddings lose nuance)
- Rerankers perform **cross-attention** between query and document
- Understands **true contextual relevance**, not just semantic proximity
- Critical for **high-stakes applications** (legal, medical, financial)

---

### Phase 6: Multi-Modal RAG (`src/08-multi_model_rag.ipynb`)

The most sophisticated pipeline — handling **real-world PDFs** with mixed content:

```
+-------------------------------------------------------------+
|                    MULTI-MODAL RAG PIPELINE                 |
+-------------------------------------------------------------+
|  INPUT: PDF (attention-is-all-you-need.pdf)                 |
|         |                                                   |
|  +-----------------------------------------------------+    |
|  |  EXTRACTION: Unstructured Library                   |    |
|  |  * Detects tables, images, text blocks              |    |
|  |  * Preserves document structure (titles, headers)   |    |
|  |         |                                           |    |
|  |  CHUNKING BY TITLE: Group elements under sections   |    |
|  |         |                                           |    |
|  |  CONTENT TYPE SEPARATION per chunk:                 |    |
|  |  |-- Images -> "Diagram showing transformer          |    |
|  |  |           architecture with multi-head attention"|    |
|  |  |-- Tables -> "Table 1: BLEU scores comparing        |    |
|  |  |           model variants on WMT 2014"           |    |
|  |  |-- Text   -> "The Transformer relies entirely      |    |
|  |  |               on attention mechanisms..."         |    |
|  |         |                                           |    |
|  |  LLM SUMMARIZATION: Generate descriptive summaries  |    |
|  |         |                                           |    |
|  |  LANGCHAIN DOCUMENTS:                               |    |
|  |  * page_content = summary (searchable, embeddable)  |    |
|  |  * metadata = {                                     |    |
|  |       original_image: base64/png,                   |    |
|  |       original_table: markdown/html,                |    |
|  |       content_type: "image|table|text",             |    |
|  |       source_section: "Architecture", ...           |    |
|  |    }                                                |    |
|  +-----------------------------------------------------+    |
|         |                                                   |
|  VECTOR STORE: Embeddings of summaries + rich metadata      |
|         |                                                   |
|  RETRIEVAL: Query -> Multi-Query -> Hybrid -> RRF -> Rerank   |
|         |                                                   |
|  GENERATION: LLM receives best chunks + original media        |
|              from metadata for grounded answers               |
+-------------------------------------------------------------+
```

---

## Project Structure

```
RAG-learning/
|-- README.md                          # This file
|-- .vscode/                           # VS Code settings
|-- data/                              # Source documents
|   |-- Google.txt                     # Company knowledge base
|   |-- Microsoft.txt
|   |-- Nvidia.txt
|   |-- SpaceX.txt
|   |-- Tesla.txt
|   |-- attention-is-all-you-need.pdf  # Multi-modal RAG target
|
|-- src/                               # Source code
|   |-- 01-ingestion_pipeline.py          # Basic RAG ingestion
|   |-- 02-retrievel_pipeline.py            # Simple retrieval + Groq generation
|   |-- 03-history_aware_generation.py      # Conversational RAG with query rewriting
|   |-- 05-character_text_splitter.py       # Character vs Recursive chunking
|   |-- 06-semantic_chunking.py             # Embedding-based semantic chunking
|   |-- 07-agentic_chunking.py              # LLM-powered intelligent chunking
|   |-- 08-multi_model_rag.ipynb            # Multi-modal PDF RAG
|   |-- 09-retrieval_methods.py             # 4 retrieval strategies
|   |-- 10-multi_query_retrieval.py         # LLM query generation + parallel retrieval
|   |-- 11-reciprocal_rank_fusion.py        # RRF scoring and deduplication
|   |-- 12-hybrid_search.ipynb              # Dense + Sparse (BM25) ensemble retrieval
|   |-- 13-reranker.ipynb                   # Cohere cross-encoder reranking
|   |-- chunks_export.json                    # Exported chunk data
|   |-- image.png                             # Demo assets
|   |-- questions.txt                         # Synthetic test questions
|   |-- rag_results.json                      # Retrieval results output
|
|-- multi-query-rag-result.json         # Sample multi-query RRF output
|-- .gitignore                          # Environment exclusions
```

---

## Installation & Setup

### Prerequisites
- Python 3.9+
- [Groq API Key](https://console.groq.com/) (for LLM inference)
- [Cohere API Key](https://cohere.com/) (for reranking — optional)
- [HuggingFace Token](https://huggingface.co/settings/tokens) (optional, for higher rate limits)

### Step 1: Clone & Install
```bash
git clone https://github.com/tahayassine-snoussi/RAG-learning.git
cd RAG-learning

python -m venv venv
source venv/bin/activate  # Windows: venv\Scriptsctivate

pip install -r requirements.txt  # Install dependencies
```

### Step 2: Environment Configuration
Create `.env` file:
```bash
GROQ_API_KEY="gsk_your_groq_api_key_here"
COHERE_API_KEY="your_cohere_api_key_here"  # Optional, for reranking
```

### Step 3: Run Pipelines
```bash
# Basic ingestion
python src/01-ingestion_pipeline.py

# Simple RAG
python src/02-retrievel_pipeline.py

# Multi-query with RRF
python src/11-reciprocal_rank_fusion.py

# Interactive history-aware chat
python src/03-history_aware_generation.py
```

---

## Usage Guide

### Quick Start: Basic RAG
```python
from src.ingestion_pipeline import load_files, chunk_documents, create_vector_store

# 1. Load
docs = load_files("./data")

# 2. Chunk
chunks = chunk_documents(docs, chunk_size=1000, chunk_overlap=200)

# 3. Store
vector_store = create_vector_store(chunks)
```

### Advanced: Multi-Query + RRF Pipeline
```python
from src.reciprocal_rank_fusion import reciprocal_rank_fusion

# After generating query variations and retrieving per query:
fused_results = reciprocal_rank_fusion(
    all_retrieved_results,  # List of doc lists per query
    k=60,                   # RRF damping factor
    verbose=True            # Detailed scoring output
)

# Top results are deduplicated and consensus-ranked
top_doc = fused_results[0][0]   # Best chunk
top_score = fused_results[0][1]  # RRF score
```

### Production: Hybrid + Rerank
```python
# From notebooks 12 & 13
hybrid_retriever = EnsembleRetriever(
    retrievers=[vector_retriever, bm25_retriever],
    weights=[0.5, 0.5]
)

# Stage 1: Get candidates
candidates = hybrid_retriever.invoke(query)

# Stage 2: Rerank for precision
reranked = reranker.compress_documents(candidates, query)
```

---

## Course Attribution

This repository was built as part of my learning journey following the excellent **YouTube course playlist**:

> **[RAG Course Playlist — Complete Guide](https://www.youtube.com/playlist?list=PLNIQLFWpQMRUMjxfe8o6g3uzJ6LH_VotY)**

The course provided the foundational concepts and structure, while this repository contains **my own implementations, extensions, and production-ready enhancements** including:
- Custom RRF implementation with detailed scoring
- Extended retrieval method comparisons
- Multi-modal pipeline architecture
- Synthetic test question generation
- Comprehensive documentation and visual pipeline diagrams

---

## Key Concepts Explained

### Reciprocal Rank Fusion (RRF)
A rank-based fusion algorithm that combines results from multiple retrieval systems without requiring score normalization. The formula:

```
RRF Score(d) = sum over all queries of [ 1 / (k + rank_q(d)) ]
```

Where `k=60` is a constant that prevents top ranks from dominating. Documents appearing in multiple result lists get exponentially boosted.

### Hybrid Search
Combines **semantic understanding** (dense vectors) with **lexical precision** (sparse BM25). The ensemble uses weighted reciprocal rank fusion to merge results from both paradigms, solving the "keyword vs. meaning" trade-off.

### Reranking
A **cross-encoder** model that jointly encodes query + document to produce a relevance score. Unlike bi-encoders (embeddings), cross-encoders capture full query-document interactions, making them 2-3x more accurate at relevance prediction — critical for the final filtering stage.

### Multi-Modal RAG
Real documents contain **tables, charts, and images** — not just text. This pipeline:
1. Extracts atomic elements using computer vision + NLP
2. Generates **LLM summaries** for non-text content (making them searchable)
3. Preserves **original media in metadata** for final answer generation
4. Enables **grounded, cited responses** with visual evidence

---

## Results & Outputs

This repository includes **demonstration outputs**:

| File | Description |
|------|-------------|
| `multi-query-rag-result.json` | Sample RRF-fused results showing financial data chunks about Tesla with metadata |
| `src/rag_results.json` | Full retrieval evaluation results |
| `src/chunks_export.json` | Exported chunk analysis data |

Example RRF output structure:
```json
{
  "chunk_id": 1,
  "enhanced_content": "The quarter ending June 2021 was the first time Tesla made a profit...",
  "metadata": {
    "original_content": {},
    "source": "Tesla.txt",
    "rrf_score": 0.0452
  }
}
```

---

## Future Enhancements

- [ ] **GraphRAG**: Knowledge graph integration for relational reasoning
- [ ] **Self-RAG**: Adaptive retrieval with reflection loops
- [ ] **RAPTOR**: Recursive tree-based chunk summarization
- [ ] **Agentic RAG**: Tool-use agents for complex multi-hop queries
- [ ] **Evaluation Framework**: RAGAS metrics for systematic quality assessment

---

## Contributing

This is a personal learning repository, but suggestions and discussions are welcome! Open an issue if you find improvements or want to discuss advanced RAG patterns.

---

## License

MIT License — feel free to use for learning and building.

---

> **Built with passion for understanding RAG deeply, not just using it superficially.**
