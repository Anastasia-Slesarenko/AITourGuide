# AI Tour Guide API Documentation

## Overview

AI Tour Guide is a REST API service for landmark recognition from images using advanced AI models including CLIP, FAISS, and Vision-Language Models (VLM).

## Base URL

```
http://{}:8000
```

## Authentication

Currently, the API does not require authentication. Rate limiting is applied per IP address.

## Rate Limiting

- **Limit**: 10 requests per 60 seconds per IP address
- **Response**: HTTP 429 (Too Many Requests) when limit exceeded
- **Header**: `Retry-After` indicates seconds until next request allowed

## Endpoints

### 1. Root Information

Get basic API information.

**Endpoint**: `GET /`

**Response**:
```json
{
  "service": "AI Tour Guide API",
  "version": "1.0.0",
  "status": "running",
  "docs": "/docs",
  "redoc": "/redoc",
  "endpoints": {
    "predict": "/v1/predict",
    "health": "/v1/health"
  }
}
```

---

### 2. Predict Landmark

Identify a landmark from an uploaded image.

**Endpoint**: `POST /v1/predict`

**Request**:
- **Content-Type**: `multipart/form-data`
- **Parameters**:
  - `image` (file, required): Image file (JPEG, PNG, GIF, WEBP)
  - `use_internet_search` (boolean, optional): Enable internet search fallback (default: true)

**Example using cURL**:
```bash
curl -X POST "http://{}:8000/v1/predict" \
  -F "image=@/path/to/landmark.jpg" \
  -F "use_internet_search=true"
```

**Example using Python**:
```python
import requests

url = "http://{}:8000/v1/predict"
files = {"image": open("landmark.jpg", "rb")}
data = {"use_internet_search": "true"}

response = requests.post(url, files=files, data=data)
print(response.json())
```

**Response** (200 OK):
```json
{
  "name": "Эйфелева башня",
  "description": "Металлическая башня в Париже, построенная в 1889 году для Всемирной выставки. Высота 330 метров.",
  "confidence": 0.95,
  "source": "retrieval",
  "timing": {
    "image_load": 0.01,
    "retrieval": 0.15,
    "vlm_generation": 2.3,
    "total": 2.46
  }
}
```

**Response Fields**:
- `name`: Identified landmark name (Russian)
- `description`: Detailed description (3-5 sentences)
- `confidence`: Confidence score (0.0 to 1.0)
- `source`: Prediction source (`retrieval`, `internet`, or `fallback`)
- `timing`: Performance breakdown in seconds

**Error Responses**:

- **400 Bad Request**: Invalid image or file too large
```json
{
  "detail": "File too large. Maximum size: 10 MB"
}
```

- **429 Too Many Requests**: Rate limit exceeded
```json
{
  "detail": "Rate limit exceeded. Try again in 45 seconds."
}
```

- **503 Service Unavailable**: Service not initialized
```json
{
  "detail": "Service not ready. AITourGuide is not initialized."
}
```

- **504 Gateway Timeout**: Prediction timeout
```json
{
  "detail": "Prediction timeout after 90 seconds"
}
```

---

### 3. Health Check

Check service health and component status.

**Endpoint**: `GET /v1/health`

**Response** (200 OK):
```json
{
  "status": "healthy",
  "service": "AITourGuide",
  "components": {
    "retriever": {
      "status": "ok",
      "index_size": 15000
    },
    "model": {
      "status": "ok",
      "device": "cuda"
    },
    "gpu": {
      "status": "ok",
      "device_name": "NVIDIA GeForce RTX 3090",
      "memory_allocated": "2.45 GB",
      "memory_reserved": "3.12 GB"
    }
  }
}
```

**Response Fields**:
- `status`: Overall status (`healthy`, `degraded`, or `unhealthy`)
- `service`: Service name
- `components`: Status of individual components

---

## Interactive Documentation

The API provides interactive documentation:

- **Swagger UI**: http://{}:8000/docs
- **ReDoc**: http://{}:8000/redoc
- **OpenAPI Schema**: http://{}:8000/openapi.json

## File Constraints

- **Maximum file size**: 10 MB
- **Supported formats**: JPEG, PNG, GIF, WEBP
- **Recommended resolution**: 224x224 to 4096x4096 pixels

## Performance

Typical response times:
- **Retrieval only**: 0.2-0.5 seconds
- **With VLM generation**: 2-5 seconds
- **With internet search**: 5-15 seconds

## Error Handling

All errors follow the standard format:
```json
{
  "detail": "Error message description"
}
```

HTTP status codes:
- `200`: Success
- `400`: Bad request (invalid input)
- `422`: Unprocessable entity (validation error)
- `429`: Rate limit exceeded
- `500`: Internal server error
- `503`: Service unavailable
- `504`: Gateway timeout

## Best Practices

1. **Image Quality**: Use clear, well-lit images for best results
2. **File Size**: Compress images before upload to reduce latency
3. **Rate Limiting**: Implement client-side rate limiting
4. **Error Handling**: Always handle timeout and service unavailable errors
5. **Caching**: Cache results for identical images to reduce API calls

## Examples

### Python Client

```python
import requests
from pathlib import Path

class AITourGuideClient:
    def __init__(self, base_url="http://{}:8000"):
        self.base_url = base_url
    
    def predict(self, image_path, use_internet_search=True):
        url = f"{self.base_url}/v1/predict"
        
        with open(image_path, "rb") as f:
            files = {"image": f}
            data = {"use_internet_search": str(use_internet_search).lower()}
            
            response = requests.post(url, files=files, data=data)
            response.raise_for_status()
            
            return response.json()
    
    def health_check(self):
        url = f"{self.base_url}/v1/health"
        response = requests.get(url)
        return response.json()

# Usage
client = AITourGuideClient()
result = client.predict("eiffel_tower.jpg")
print(f"Landmark: {result['name']}")
print(f"Confidence: {result['confidence']:.2%}")
```

### JavaScript/Node.js Client

```javascript
const FormData = require('form-data');
const fs = require('fs');
const axios = require('axios');

async function predictLandmark(imagePath, useInternetSearch = true) {
  const form = new FormData();
  form.append('image', fs.createReadStream(imagePath));
  form.append('use_internet_search', useInternetSearch);
  
  try {
    const response = await axios.post(
      'http://{}:8000/v1/predict',
      form,
      { headers: form.getHeaders() }
    );
    
    return response.data;
  } catch (error) {
    console.error('Error:', error.response?.data || error.message);
    throw error;
  }
}

// Usage
predictLandmark('eiffel_tower.jpg')
  .then(result => {
    console.log(`Landmark: ${result.name}`);
    console.log(`Confidence: ${(result.confidence * 100).toFixed(2)}%`);
  });
```

## Support

For issues and questions:
- GitHub Issues: https://github.com/Anastasia-Slesarenko/AITourGuide/issues
- Documentation: https://github.com/Anastasia-Slesarenko/AITourGuide/docs
