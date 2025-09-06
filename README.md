# RAG System

A complete end-to-end **RAG (Retrieval-Augmented Generation)** system built with **ChromaDB** vector database, featuring document processing, semantic search, and evaluation capabilities.

## Features

### Core RAG Pipeline

- **Document Processing**: PDF, YouTube transcripts, websites, text input
- **Vector Database**: ChromaDB for storing and searching document embeddings
- **Semantic Search**: OpenAI embeddings with cosine similarity
- **Response Generation**: OpenAI GPT-3.5 with retrieved context
- **Web Search Integration**: Real-time information via Serper API

### RAG Evaluation (RAGAS)

- **Faithfulness**: Measures answer grounding in retrieved context
- **Answer Relevancy**: Evaluates answer relevance to questions
- **Context Precision**: Assesses quality of retrieved context
- **Performance Metrics**: Response time and retrieval statistics

## Project Structure

```
rag-system/
│
├── main.py                 # Main Streamlit application
├── config.py              # Configuration settings
├── vector_database.py     # ChromaDB vector database operations
├── document_processor.py  # Document extraction and chunking
├── web_search.py          # Web search functionality
├── rag_system.py          # Complete RAG pipeline
├── rag_evaluation.py      # RAGAS evaluation framework
├── requirements.txt       # Python dependencies
└── README.md             # This file
```

## Setup Instructions

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Get API Keys

- **OpenAI API Key** (Required): [Get here](https://platform.openai.com/api-keys)
- **Serper API Key** (Optional): [Get here](https://serper.dev) for web search

### 3. Run the Application

```bash
streamlit run main.py
```

### 4. Configure in Sidebar

- Enter your OpenAI API key
- Optionally add Serper API key for web search
- Click "Initialize RAG System"

## How to Use

### 1. Add Documents to Knowledge Base

- **PDFs**: Upload multiple PDF files
- **YouTube**: Paste YouTube URLs for transcript extraction
- **Websites**: Enter website URLs for content scraping
- **Text**: Direct text input with custom titles

### 2. Chat with Your Knowledge Base

- Ask questions about uploaded documents
- Toggle web search for real-time information
- View sources used for each response

### 3. Evaluate RAG Performance

- Create test questions
- Run RAGAS evaluation metrics
- Analyze faithfulness, relevancy, and precision scores
- Download evaluation results

## Understanding Vector Databases

### What is ChromaDB?

ChromaDB is an open-source vector database that:

- Stores high-dimensional vectors (embeddings) representing text semantics
- Performs fast similarity search using cosine distance
- Uses HNSW algorithm for efficient approximate nearest neighbor search
- Manages collections of documents with metadata

### How Vector Search Works

1. **Text → Embeddings**: Convert documents to 1536-dimensional vectors using OpenAI
2. **Storage**: Store embeddings in ChromaDB collections
3. **Query Processing**: Convert user questions to embedding vectors
4. **Similarity Search**: Find closest vectors using cosine distance
5. **Context Retrieval**: Return most semantically similar document chunks

## RAG Pipeline Explained

### 1. Retrieval Phase

- User asks a question
- Question is converted to embedding vector
- ChromaDB performs similarity search
- Most relevant document chunks are retrieved

### 2. Augmentation Phase  

- Retrieved context is combined with user question
- System prompt is constructed with context
- Additional web search results (if enabled)

### 3. Generation Phase

- OpenAI GPT-3.5 generates response using context
- Response is grounded in retrieved documents
- Sources are tracked for attribution

## RAGAS Evaluation Metrics

### Faithfulness

Measures how well the generated answer is supported by the retrieved context:

- **High score**: Answer claims are backed by context
- **Low score**: Answer contains unsupported information

### Answer Relevancy

Evaluates how well the answer addresses the user's question:

- **High score**: Direct, relevant response to question
- **Low score**: Off-topic or incomplete answer

### Context Precision

Assesses the quality of retrieved context:

- **High score**: Retrieved chunks are highly relevant
- **Low score**: Irrelevant chunks retrieved

## 🛠️ Customization

### Adding New Document Types

1. Add extraction function in `document_processor.py`
2. Update UI in `main.py` knowledge base tab
3. Test with sample documents

### Modifying Vector Search

- Adjust `CHUNK_SIZE` and `CHUNK_OVERLAP` in `config.py`
- Change `SIMILARITY_SEARCH_RESULTS` for more/fewer results
- Experiment with different distance functions in ChromaDB

### Custom Evaluation Metrics

- Extend `RAGEvaluator` class in `rag_evaluation.py`
- Add new metrics for domain-specific evaluation
- Integrate with full RAGAS framework for advanced metrics

## Vector Database Concepts

### Embeddings

- **Definition**: Numerical representations of text semantics
- **Dimensions**: 1536-dimensional vectors from OpenAI text-embedding-3-small
- **Similarity**: Similar texts have similar vector representations

### Collections

- **Purpose**: Organize related documents in ChromaDB
- **Metadata**: Store source information, timestamps, chunk indices
- **Indexing**: HNSW algorithm for fast approximate search

### Distance Functions

- **Cosine Distance**: Measures angle between vectors (good for text)
- **Euclidean Distance**: Measures straight-line distance
- **Dot Product**: Measures vector alignment
