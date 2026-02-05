# AI-Generated Voice Detection API

FastAPI backend for detecting AI-generated speech vs human speech. Supports **Tamil**, **English**, **Hindi**, **Malayalam**, and **Telugu** languages.

Uses **AASIST ensemble inference** (original + fine-tuned models) with heuristic pattern rules for improved multilingual detection. Built for production deployment on **Google Cloud Run (CPU-only)**.

## Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Configure Environment

```bash
# Copy example config
cp .env.example .env

# Edit .env and set your API key and model paths
# VOICE_API_KEYS=your-secret-key-here
# AASIST_ORIG_WEIGHTS_GCS_URI=gs://bucket/original.pth
# AASIST_FT_WEIGHTS_GCS_URI=gs://bucket/finetuned.pth
```

### 3. Run Locally

```bash
# Start the server
python api.py

# Or use uvicorn directly
uvicorn api:app --host 0.0.0.0 --port 8001 --reload
```

Server runs at `http://localhost:8001` with docs at `/docs` (when `ENABLE_DOCS=1`).

### 4. Test the API

```bash
# Python test script
python scripts/test_local.py --audio your_audio.mp3 --api-key your-key

# Or curl
./scripts/test_curl.sh your_audio.mp3 your-key
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
  "explanation": "Ensemble: English FAKE (direct detection). Orig(fake=93%), FT(real=45%)"
}
```

### Error Response (4xx/5xx)

```json
{
  "status": "error",
  "message": "Invalid base64 encoding in audioBase64 field."
}
```

---

## Configuration

| Variable                     | Default                          | Description                             |
| ---------------------------- | -------------------------------- | --------------------------------------- |
| `VOICE_API_KEYS`             | (required)                       | Comma-separated valid API keys          |
| `AASIST_DEVICE`              | `cpu`                            | Device: `cpu` or `cuda`                 |
| `AASIST_ORIG_WEIGHTS_GCS_URI`| (empty)                          | GCS URI for original model weights      |
| `AASIST_FT_WEIGHTS_GCS_URI`  | (empty)                          | GCS URI for fine-tuned model weights    |
| `AASIST_ORIG_CACHE_PATH`     | `/tmp/aasist_original.pth`       | Local cache path for original weights   |
| `AASIST_FT_CACHE_PATH`       | `/tmp/aasist_finetuned_best.pth` | Local cache path for finetuned weights  |
| `AASIST_THRESHOLD`           | `0.5`                            | Classification threshold                |
| `AASIST_MAX_WINDOWS`         | `3`                              | Max windows for long audio              |
| `ENABLE_DOCS`                | `0`                              | Set to `1` to enable `/docs`            |
| `MAX_MP3_BYTES`              | `15000000`                       | Max decoded MP3 size (15 MB)            |
| `MAX_DURATION_SECONDS`       | `300`                            | Max audio duration (5 minutes)          |

---

## Ensemble Model Architecture

This API uses an **ensemble of two AASIST models**:

1. **Original AASIST** - English-trained baseline
2. **Fine-tuned AASIST** - Multilingual fine-tuned model

### Multi-Window Inference

For audio longer than ~4 seconds, the detector samples multiple evenly-spaced windows (default: 3) and averages scores.

### Heuristic Pattern Rules

The ensemble uses pattern detection to handle inverted predictions for Indian languages:

| Pattern | Original Model | Fine-tuned | Result |
|---------|---------------|------------|--------|
| Indian REAL | >95% fake | >95% real | HUMAN (inverted) |
| Indian FAKE | <15% fake | >80% real | AI_GENERATED (inverted) |
| English FAKE | >92% fake | - | AI_GENERATED |
| English REAL | 85-92% fake | >95% real | HUMAN |

---

## Deploy to Cloud Run (Recommended)

### Prerequisites

1. **Install and authenticate gcloud CLI**:
   ```bash
   gcloud auth login
   gcloud config set project voice-detect-cloudrun
   ```

2. **Set your API key** in `.env` or environment:
   ```bash
   export VOICE_API_KEYS=your-secret-api-key
   ```

3. **Model weights** are pre-configured to load from GCS bucket:
   - `gs://voice-detect-168345068797-models/models/aasist_original.pth`
   - `gs://voice-detect-168345068797-models/models/aasist_finetuned_best.pth`

### Deploy

```bash
bash scripts/deploy_cloudrun.sh
```

This script will:
- Validate prerequisites (gcloud auth, project, API key)
- Build image via Cloud Build
- Push to Artifact Registry (`asia-south1-docker.pkg.dev`)
- Deploy to Cloud Run with optimized settings
- Print the service URL

### Test the Deployment

```bash
# Get service URL
export SERVICE_URL=$(gcloud run services describe voice-detect-api --region asia-south1 --format='value(status.url)')

# Test with sample audio
bash scripts/test_curl.sh sample.mp3
```

### Resource Allocation

| Setting | Value | Reason |
|---------|-------|--------|
| Memory | 4 GiB | Two PyTorch models |
| CPU | 2 vCPUs | Ensemble inference |
| Timeout | 300s | Long audio processing |
| Concurrency | 1 | ML model isolation |
| Max instances | 2 | Cost control |

**Note**: Cloud Run injects `PORT` env var. Container listens on `0.0.0.0:${PORT}`.

---

## Testing

```bash
# Run all tests
pytest tests/ -v

# Tests mock the detector - no model files needed
```

---

## Project Structure

```
├── api.py                 # FastAPI app entrypoint
├── config.py              # Settings (pydantic-settings)
├── platform_services.py   # /api/voice-detection endpoint
├── models/
│   └── AASIST.py          # AASIST model architecture
├── services/
│   ├── audio_io.py        # FFmpeg MP3 decoder
│   ├── gcs_weights.py     # GCS download utility
│   └── qc.py              # Quality control metrics
├── detectors/
│   ├── base.py            # BaseDetector interface
│   ├── registry.py        # Detector initialization
│   └── aasist_detector.py # Ensemble detector
├── scripts/
│   ├── deploy_cloudrun.sh # Cloud Run deployment
│   ├── test_local.py      # Python test script
│   └── test_curl.sh       # Curl test script
├── tests/
├── Dockerfile
├── requirements.txt
└── .env.example
```

---

## License

MIT
