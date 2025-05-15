# BetrCache: Multimodal Semantic Cache for LLM Applications

BetrCache is a semantic caching system tailored specifically to multimodal inputs (text and images), designed to significantly reduce latency and API costs associated with querying large language models (LLMs). It leverages CLIP embeddings and approximate nearest neighbor (ANN) indexing to efficiently cache and retrieve semantically similar responses.

## System Overview

BetrCache supports:

- **Text-only caching**: For purely textual inputs using CLIP text embeddings.
- **Multimodal caching**: For combined text-image queries, using joint CLIP embeddings.

The system integrates Redis as the cache storage backend and employs Hnswlib’s ANN indexing (`HnswAnnIndex`) for efficient similarity searches.

## Codebase Structure

```
src/
├── ann_index.py # ANN indexing and search logic
├── api.py # LLM queries and embedding generation
├── cache.py # Redis-backed embedding caches
├── cache_client.py # Redis client wrapper
├── config.py # Configuration parameters
├── custom_types.py # Data models and schemas
├── dataset.py # Flickr30k dataset loader
├── judge.py # Similarity scoring for responses
├── similarity.py # Cosine similarity implementations
└── utils.py # Utility and logging functions

main.py # Interactive REPL interface
evaluate.py # Evaluation script
plotting.py # Plotting evaluation results
```


## Setup

### 1. Clone the repository
```bash
git clone git@github.com:AydanPirani/BetrCache.git
cd BetrCache
```

### 2. Install dependencies
Ensure that Python 3.10 is installed. Then, open the venv and install the required Python packages:
```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### 3. Set up environment variables

Create a `.env` file in the project root and add the following environment variables:

```
REDIS_URL=<your-redis-instance>
OPENAI_API_KEY=<your-openai-api-key>
LLM_MODEL=gpt-4.1
EMBEDDINGS_MODEL=text-embedding-3-small
TEXT_EMBEDDING_DIMENSION=1536
IMAGE_EMBEDDING_DIMENSION=512
THRESHOLD=5
SIMILARITY_THRESHOLD=0.8
```

### 4. Run the REPL

To start the interactive REPL, run:

```bash
python main.py
```

### 5. Run the evaluation script

To evaluate the system's performance, run:

```bash
python evaluation.py
```

We provide some example samples to run evaluations with. For additional use cases, download the Flickr10k dataset and modify the `get_dataset` function in `src/dataset.py` to load your desired dataset.

## Key Configuration Options
- `THRESHOLD`: Number of ANN candidates retrieved per query.
- `SIMILARITY_THRESHOLD`: Cosine similarity threshold to determine cache hits.
- Embedding dimensions based on selected embedding models.