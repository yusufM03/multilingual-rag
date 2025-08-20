# Multilingual RAG

## Table of Contents

* [🎥 Demo Video](#🎥-demo-video)
* [🏗 Architecture Overview](#🏗-architecture-overview)

  * [1. Document Ingestion & Extraction](#1-document-ingestion--extraction)
  * [2. Embedding & Vector Storage](#2-embedding--vector-storage)
  * [3. Query Processing & Retrieval](#3-query-processing--retrieval)
  * [4. RAG Generation Layer](#4-rag-generation-layer)
  * [5. User Interface & Deployment](#5-user-interface--deployment)
* [🚀 Quick Start](#🚀-quick-start)

  * [Online Demo](#online-demo)
  * [1. Prerequisites](#1-prerequisites)
  * [2. Environment Configuration](#2-environment-configuration)
  * [2. Running with Docker](#2-running-with-docker)
  * [3. Running Locally (From Source)](#3-running-locally-from-source)
  * [7. Project Structure](#7-project-structure)
  * [9. LLMOps Evaluation](#9-llmops-evaluation)

---

## 🎥 Demo Video

[![Watch the demo](docs/screen.png)](https://www.youtube.com/watch?v=UE_m3gu2TEE)

---

## 🏗 Architecture Overview

The **Multilingual RAG System** is designed for **Arabic and English document understanding**.
It separates **document processing, storage, and querying** into clear layers to ensure **scalability, multilingual support, and high-speed retrieval**.

![Architecture Overview](docs/overview.png)

### 1. Document Ingestion & Extraction

* Converts raw PDFs/DOCs into structured text blocks with metadata.
* Supports **complex layouts** (tables, multi-column text, footnotes).
* Automatically detects **Arabic (RTL)** vs. **English (LTR)** content.
* Outputs **clean, chunked text** ready for embedding.

### 2. Embedding & Vector Storage

* Converts text chunks into **multilingual semantic embeddings**.
* Stores vectors in **Qdrant Cloud** with:

  * **Language-specific collections** (Arabic / English)
  * **High-speed vector similarity search**
  * **Cloud scalability for large datasets**

### 3. Query Processing & Retrieval

* Detects query language automatically.
* Retrieves **top semantic matches** from the relevant collection.
* Aggregates context for accurate, language-specific responses.

### 4. RAG Generation Layer

* Powered by **Groq-accelerated Llama models**.
* Produces **multilingual answers** enriched with:

  * **Contextual reasoning**
  * **Source citations** (page numbers & document names)

### 5. User Interface & Deployment

* **Streamlit web app** for interactive queries and document uploads.
* **Streamlit Cloud deployment** with secure secrets management.

---

## 🚀 Quick Start

### Online Demo

Try the live app:
[https://multilingual-rag-app.streamlit.app/](https://multilingual-rag-app.streamlit.app/)

---

### 1. Prerequisites

* Docker installed
* A free **Qdrant Cloud** account
* API keys for:

  * **Groq** (LLM)
  * **Qdrant** (vector DB)
  * **LlamaCloud** (PDF extraction)

**Required API Keys**

* **Qdrant Cloud (Vector Database)**
  Go to Qdrant Cloud → Create a free account → Create a new cluster → Copy your cluster URL and API key

* **Groq (LLM Provider)**
  Visit Groq Console → Sign up and create API keys
  (We use two keys for load balancing Arabic and English)

* **LlamaCloud (PDF Extraction)**
  Go to LlamaCloud → Sign up and get your API key
  (Required for document parsing)

---

### 2. Environment Configuration

Create a `.streamlit/secrets.toml` file in the **project root** with your API keys:

```toml
# Qdrant Cloud Configuration
qdrant_url=https://your-cluster-url.qdrant.tech:6333
qdrant_api_key=your-qdrant-api-key

# Groq API Keys (for LLM)
groq_api_key=gsk_your-groq-api-key-for-arabic
groq_api_key_1=gsk_your-groq-api-key-for-english

# LlamaCloud API Key (for PDF extraction)
llama_cloud_api_key=llx_your-llamacloud-api-key
```

---

# 🚀 Running the RAG App

## 1. Running with Docker (Prebuilt Image)

You can pull and run the prebuilt image from Docker Hub directly.

### **Windows PowerShell**

```powershell
# Run prebuilt image from Docker Hub with secrets mounted
docker run --rm -p 8501:8501 `
  -v ${PWD}/.streamlit/secrets.toml:/app/.streamlit/secrets.toml `
  ymack/my-rag-app:latest
```

### **Linux / macOS**

```bash
docker run --rm -p 8501:8501 \
  -v $(pwd)/.streamlit/secrets.toml:/app/.streamlit/secrets.toml \
  ymack/my-rag-app:latest
```

Then open your browser to:
[http://localhost:8501](http://localhost:8501)

---

## 2. Running Locally (From Source)

If you prefer to build and run the image locally:

```bash
# Clone the repository
git clone https://github.com/yusufM03/multilingual-rag.git
cd multilingual-rag

# Build the Docker image locally
docker build -t my-rag-app .

# Run the container with secrets mounted
docker run --rm -p 8501:8501 \
  -v $(pwd)/.streamlit/secrets.toml:/app/.streamlit/secrets.toml \
  my-rag-app
```

---

Make sure `.streamlit/secrets.toml` exists **before** running the container.



### 7. Project Structure

```
yusufm03-multilingual-rag//
├── .streamlit/
│   └── secrets.toml
├── config/
│   ├── __init__.py
│   └── settings.py
├── docs/
│   ├── Agentic RAG Task.pdf
│   ├── overview.png
├── Documents/
├── models/
│   └── rag_response.py
├── src/
│   ├── __init__.py
│   ├── app.py
│   └── core.py
├── utils/
│   ├── language_detector.py
│   └── store_logs.py
├── .dockerignore
├── .gitignore
├── dockerfile
├── readme.md
└── requirements.txt
```

---

### 9. LLMOps Evaluation

Integrated with Weights & Biases (W\&B) Cloud for:

* Version control of prompts & models
* Retrieval monitoring & drift detection
* Feedback-driven optimization
* Regression testing & batch evaluation

```
User Query → Qdrant Retrieval → LLM Response
       ↓                   ↓
  Retrieve Top-K        Compute Evaluation
   (context)        ┌──────────────┐
        └─────────▶│ Auto Metrics │
                   │ + W&B Logging│
                   └──────────────┘
                           ↓
              Continuous Monitoring & Improvement
                           ↓
                 Batch Regression Testing
                           ↓
               Model/Prompt Updates & Deployment
```
