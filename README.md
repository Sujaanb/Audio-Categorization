# AI-Generated Voice Detection API

FastAPI backend for detecting AI-generated speech vs human speech. Supports **Tamil**, **English**, **Hindi**, **Malayalam**, and **Telugu** languages.

Uses the **AASIST model** for inference.

## Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Configure Environment

```bash
# Copy example config
cp .env.example .env

# Edit .env and set your API key
# VOICE_API_KEYS=your-secret-key-here
```

### 3. Add Model Weights

Place your AASIST model weights file at:
```
model_weights/aasist_original.pth
```

### 4. Run Locally

```bash
# Start the server
python api.py

# Or use uvicorn directly
uvicorn api:app --host 0.0.0.0 --port 8001 --reload
```

Server runs at `http://localhost:8001` with docs at `/docs` (when `ENABLE_DOCS=1`).

### 5. Test the API

Use the Swagger UI at `http://localhost:8001/docs` or send a request:

```bash
curl -X POST http://localhost:8001/api/voice-detection \
  -H "Content-Type: application/json" \
  -H "x-api-key: your-key" \
  -d '{"language":"English","audioFormat":"mp3","audioBase64":"<base64-encoded-mp3>"}'
```

---

## API Reference

### Endpoint

```
POST /api/voice-detection
```

### Headers

| Header         | Required | Description                         |
| -------------- | -------- | ----------------------------------- |
| `Content-Type` | Yes      | Must be `application/json`          |
| `x-api-key`    | Yes      | Your API key (case-insensitive)     |

### Request Body

```json
{
  "language": "English",
  "audioFormat": "mp3",
  "audioBase64": "<Base64-encoded MP3 bytes>"
}
```

| Field         | Type   | Description                                                |
| ------------- | ------ | ---------------------------------------------------------- |
| `language`    | string | One of: `Tamil`, `English`, `Hindi`, `Malayalam`, `Telugu` |
| `audioFormat` | string | Only `mp3` (case-insensitive)                              |
| `audioBase64` | string | Base64-encoded MP3 file                                    |

### Success Response (200)

```json
{
  "status": "success",
  "language": "English",
  "classification": "AI_GENERATED",
  "confidenceScore": 0.87,
  "explanation": "Detected AI-generated speech. Spoof probability: 87%"
}
```

**Response Headers:**
```
RateLimit-Limit: 100
RateLimit-Remaining: 99
RateLimit-Reset: 1681234560
```

### Error Response (4xx/5xx)

```json
{
  "status": "error",
  "message": "Invalid base64 encoding in audioBase64 field."
}
```

### Rate Limited Response (429)

```json
{
  "status": "error",
  "message": "Rate limit exceeded. Maximum 100 requests per 60s."
}
```

**Response Headers:**
```
RateLimit-Limit: 100
RateLimit-Remaining: 0
RateLimit-Reset: 1681234560
```

---

## Rate Limiting

### Overview

The API enforces rate limiting on a **per-API-key basis** to ensure fair resource usage and prevent abuse.

- **Default Limit:** 100 requests per 60 seconds per API key
- **Response Code:** 429 Too Many Requests
- **Rate Limit Headers:** Include `RateLimit-*` headers in all responses

### Headers

All responses include rate limit information:

| Header              | Description                                  |
| ------------------- | -------------------------------------------- |
| `RateLimit-Limit`   | Maximum requests per period (100)            |
| `RateLimit-Remaining` | Requests remaining in current window       |
| `RateLimit-Reset`   | Unix timestamp when window resets            |

### Configuration

| Variable                | Default | Description                                    |
| ----------------------- | ------- | ---------------------------------------------- |
| `RATE_LIMIT_REQUESTS`   | `100`   | Max requests per period per API key            |
| `RATE_LIMIT_PERIOD_SECONDS` | `60` | Time window in seconds for rate limit         |

**Example:** Set stricter limits in `.env`:
```bash
RATE_LIMIT_REQUESTS=50
RATE_LIMIT_PERIOD_SECONDS=60
```

### Handling Rate Limits

When you receive a 429 response:

1. Check the `RateLimit-Reset` header for the Unix timestamp when your quota resets
2. Wait until that time before retrying
3. Implement exponential backoff with jitter for robust clients

Example retry logic (Python):
```python
import time
import requests

def detect_with_retry(audio_base64, api_key, max_retries=3):
    for attempt in range(max_retries):
        response = requests.post(
            "http://localhost:8001/api/voice-detection",
            json={"language": "English", "audioFormat": "mp3", "audioBase64": audio_base64},
            headers={"x-api-key": api_key}
        )
        
        if response.status_code == 429:
            reset_time = int(response.headers["RateLimit-Reset"])
            wait_seconds = max(0, reset_time - time.time())
            print(f"Rate limited. Waiting {wait_seconds:.1f}s...")
            time.sleep(wait_seconds + 1)
            continue
        
        return response.json()
    
    raise Exception("Max retries exceeded")
```

---

## Metrics & Analytics

### Get Your Metrics

Access usage statistics for your API key:

```bash
curl -X GET http://localhost:8001/api/admin/metrics \
  -H "x-api-key: your-key"
```

**Response (when `ENABLE_METRICS_ENDPOINT=1`):**
```json
{
  "status": "success",
  "metrics": {
    "total_requests": 250,
    "success_requests": 248,
    "failed_requests": 2,
    "success_rate": 0.992,
    "average_latency_ms": 1250.5,
    "languages": {
      "English": 150,
      "Tamil": 100
    }
  }
}
```

### Configuration

| Variable | Default | Description |
| -------- | ------- | ----------- |
| `ENABLE_METRICS_ENDPOINT` | `0` | Set to `1` to enable `/api/admin/metrics` |

Enable metrics in `.env`:
```bash
ENABLE_METRICS_ENDPOINT=1
```

---

## Configuration

| Variable                | Default                              | Description                             |
| ----------------------- | ------------------------------------ | --------------------------------------- |
| `VOICE_API_KEYS`        | (required)                           | Comma-separated valid API keys          |
| `AASIST_DEVICE`         | `cpu`                                | Device: `cpu` or `cuda`                 |
| `AASIST_WEIGHTS_PATH`   | `./model_weights/aasist_original.pth`| Path to model weights file              |
| `AASIST_THRESHOLD`      | `0.5`                                | Classification threshold                |
| `AASIST_MAX_WINDOWS`    | `3`                                  | Max windows for long audio              |
| `ENABLE_DOCS`           | `0`                                  | Set to `1` to enable `/docs`            |
| `MAX_MP3_BYTES`         | `15000000`                           | Max decoded MP3 size (15 MB)            |
| `MAX_DURATION_SECONDS`  | `300`                                | Max audio duration (5 minutes)          |
| `RATE_LIMIT_REQUESTS`   | `100`                                | Max requests per period per API key     |
| `RATE_LIMIT_PERIOD_SECONDS` | `60`                            | Rate limit period in seconds            |
| `ENABLE_METRICS_ENDPOINT` | `0`                                | Enable `/api/admin/metrics` endpoint    |

---

## Multi-Window Inference

For audio longer than ~4 seconds, the detector samples multiple evenly-spaced windows (default: 3) and averages scores.

---

## Project Structure

```
├── api.py                 # FastAPI app entrypoint
├── config.py              # Settings (pydantic-settings)
├── platform_services.py   # /api/voice-detection endpoint
├── base_requests.py       # Request/response models
├── models/
│   └── AASIST.py          # AASIST model architecture
├── services/
│   ├── audio_io.py        # Audio decoder
│   ├── qc.py              # Quality control metrics
│   └── rate_limiter.py    # Rate limiting & metrics tracking (NEW)
├── detectors/
│   ├── base.py            # BaseDetector interface
│   ├── registry.py        # Detector initialization
│   └── aasist_detector.py # AASIST detector
├── model_weights/
│   └── aasist_original.pth # Model weights (not in repo)
├── audio_samples/         # Sample audio files
├── requirements.txt
└── .env.example
```

---

## License

MIT
