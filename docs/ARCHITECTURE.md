# AI Tour Guide Architecture

## System Overview

AI Tour Guide is a landmark recognition service that combines multiple AI technologies:

1. **CLIP** (Contrastive Language-Image Pre-training) for image encoding
2. **FAISS** (Facebook AI Similarity Search) for fast vector search
3. **VLM** (Vision-Language Model) for generating descriptions
4. **Internet Search** (Yandex + Wikipedia) as fallback

## Architecture Diagram

```
┌─────────────────────────────────────────────────────────────────┐
│                         Client Application                       │
└────────────────────────────┬────────────────────────────────────┘
                             │ HTTP/REST
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│                        FastAPI Server                            │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐          │
│  │ Rate Limiter │  │     CORS     │  │  Validation  │          │
│  └──────────────┘  └──────────────┘  └──────────────┘          │
└────────────────────────────┬────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│                      AITourGuide Service                         │
│                                                                   │
│  ┌───────────────────────────────────────────────────────────┐  │
│  │                    Prediction Pipeline                     │  │
│  │                                                             │  │
│  │  1. Image Load & Validation                                │  │
│  │           ▼                                                 │  │
│  │  2. CLIP Encoding (768-dim vector)                         │  │
│  │           ▼                                                 │  │
│  │  3. FAISS Search (top-10 candidates)                       │  │
│  │           ▼                                                 │  │
│  │  4. VLM Generation (with RAG context)                      │  │
│  │           ▼                                                 │  │
│  │  5. Confidence Calculation                                 │  │
│  │           ▼                                                 │  │
│  │  6. Internet Search (if confidence < threshold)            │  │
│  │           ▼                                                 │  │
│  │  7. Response Formatting                                    │  │
│  └───────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────┘
                             │
        ┌────────────────────┼────────────────────┐
        ▼                    ▼                    ▼
┌──────────────┐    ┌──────────────┐    ┌──────────────┐
│ CLIP Model   │    │ FAISS Index  │    │  VLM Model   │
│ (openai/     │    │ (15K vectors)│    │ (Qwen2-VL)   │
│  clip-vit-   │    │              │    │              │
│  large)      │    │              │    │              │
└──────────────┘    └──────────────┘    └──────────────┘
```

## Component Details

### 1. FastAPI Server

**Location**: `main.py`, `src/api/`

**Responsibilities**:
- HTTP request handling
- Input validation
- Rate limiting
- CORS management
- Error handling
- Response formatting

**Key Features**:
- Async/await for concurrent requests
- Dependency injection for service management
- Automatic OpenAPI documentation
- Middleware for cross-cutting concerns

### 2. AITourGuide Service

**Location**: `services/ai_tour_guide.py`

**Responsibilities**:
- Orchestrating the prediction pipeline
- Managing model lifecycle
- Calculating confidence scores
- Triggering internet search fallback
- Performance metrics tracking

**Key Methods**:
- `predict()`: Main prediction method
- `_calculate_confidence()`: Confidence scoring
- `_search_internet()`: Fallback search
- `health_check()`: Service health status

### 3. RAG Retriever

**Location**: `rag/retriever.py`

**Responsibilities**:
- CLIP model management
- Image encoding
- FAISS index search
- Facts database lookup
- Context formatting

**Key Features**:
- L2-normalized embeddings
- Cosine similarity search
- Batch processing support
- Optional multimodal reranking

### 4. VLM (Vision-Language Model)

**Location**: Integrated in `services/ai_tour_guide.py`

**Model**: Qwen2-VL-2B with LoRA adapter

**Responsibilities**:
- Generating landmark descriptions
- Processing image + text prompts
- JSON response formatting

**Input Format**:
```python
{
    "role": "user",
    "content": [
        {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64,..."}},
        {"type": "text", "text": "Prompt with RAG context"}
    ]
}
```

### 5. Internet Search

**Location**: `services/yandex_search.py`, `services/translator.py`

**Components**:
- **Yandex Image Search**: Reverse image search
- **Wikipedia Service**: Fetch landmark information
- **Yandex Translator**: Translate to Russian

**Workflow**:
1. Upload image to Yandex Vision API
2. Extract landmark names from search results
3. Query Wikipedia for each name
4. Translate content to Russian
5. Return best match

## Data Flow

### Successful Retrieval Path

```
Image Upload
    ↓
CLIP Encoding (0.1s)
    ↓
FAISS Search (0.05s)
    ↓
VLM Generation (2-3s)
    ↓
Confidence: 0.85 (> 0.78 threshold)
    ↓
Return Result
```

### Internet Search Fallback Path

```
Image Upload
    ↓
CLIP Encoding (0.1s)
    ↓
FAISS Search (0.05s)
    ↓
VLM Generation (2-3s)
    ↓
Confidence: 0.65 (< 0.78 threshold)
    ↓
Yandex Image Search (2-5s)
    ↓
Wikipedia Lookup (1-3s)
    ↓
Translation (0.5-1s)
    ↓
Return Result
```

## Confidence Calculation

The confidence score combines multiple signals:

```python
confidence = (
    0.25 * clip_score +           # CLIP similarity
    0.20 * gap_score +            # Gap between top-2 candidates
    0.35 * name_match_score +     # Name matching
    0.20 * position_score         # Position in retrieved list
)
```

**Thresholds**:
- `>= 0.78`: High confidence, return immediately
- `< 0.78`: Low confidence, trigger internet search

## Performance Optimization

### 1. Model Loading
- Models loaded once at startup
- Kept in memory for fast inference
- GPU acceleration when available

### 2. Caching
- FAISS index loaded once
- Facts database cached in memory
- No per-request model loading

### 3. Async Processing
- Non-blocking I/O operations
- Concurrent request handling
- Timeout protection

### 4. Resource Management
- Automatic GPU memory cleanup
- Context managers for services
- Graceful shutdown handling

## Scalability Considerations

### Current Limitations
- Single instance (no horizontal scaling)
- In-memory rate limiting (not distributed)
- Synchronous model inference

### Future Improvements
1. **Horizontal Scaling**:
   - Load balancer (nginx/HAProxy)
   - Shared Redis for rate limiting
   - Distributed caching

2. **Model Optimization**:
   - Model quantization (INT8/INT4)
   - Batch inference
   - Model serving (TorchServe/TensorRT)

3. **Database**:
   - PostgreSQL for facts database
   - Vector database (Milvus/Qdrant)
   - Caching layer (Redis)

4. **Monitoring**:
   - Prometheus metrics
   - Grafana dashboards
   - Distributed tracing (Jaeger)

## Security

### Current Measures
- File size limits (10 MB)
- File type validation
- Rate limiting per IP
- Input sanitization

### Recommendations
1. Add authentication (API keys/JWT)
2. Implement request signing
3. Add HTTPS/TLS
4. Content Security Policy headers
5. DDoS protection (Cloudflare)

## Deployment

### Development
```bash
make run
# or
uvicorn main:app --reload
```

### Production
```bash
# Docker
docker-compose up -d

# Or direct
uvicorn main:app --host 0.0.0.0 --port 8000 --workers 4
```

### Environment Variables
See `.env.example` for configuration options.

## Monitoring

### Health Checks
- Endpoint: `GET /v1/health`
- Checks: Model loaded, retriever ready, GPU available

### Metrics
- Total requests
- Success/failure rate
- Average confidence
- Internet search rate
- Response times (retrieval, generation, total)

### Logging
- Structured logging (JSON format)
- Log levels: DEBUG, INFO, WARNING, ERROR
- Correlation IDs for request tracing

## Technology Stack

| Component | Technology | Version |
|-----------|-----------|---------|
| Web Framework | FastAPI | 0.109+ |
| ML Framework | PyTorch | 2.2.2 |
| Vision Model | CLIP | openai/clip-vit-large-patch14 |
| VLM | Qwen2-VL | 2B with LoRA |
| Vector Search | FAISS | 1.9.0 |
| Server | Uvicorn | 0.27+ |
| Python | CPython | 3.10+ |

## References

- [CLIP Paper](https://arxiv.org/abs/2103.00020)
- [FAISS Documentation](https://github.com/facebookresearch/faiss)
- [FastAPI Documentation](https://fastapi.tiangolo.com/)
- [Qwen2-VL](https://github.com/QwenLM/Qwen2-VL)
