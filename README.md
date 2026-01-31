# AI-Generated Voice Detection API

FastAPI backend for detecting AI-generated speech vs human speech. Supports **Tamil**, **English**, **Hindi**, **Malayalam**, and **Telugu** languages.

Built for the hackathon "AI-Generated Voice Detection" with production deployment on Google Cloud Run.

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

### 3. Run Locally

```bash
# Option 1: Direct Python
python api.py

# Option 2: Use the script (Linux/Mac)
chmod +x scripts/run_local.sh
./scripts/run_local.sh
```

Server runs at `http://localhost:8001` with docs at `/docs` (when `ENABLE_DOCS=1`).

---

## API Reference

### Endpoint

```
POST /api/voice-detection
```

### Headers

| Header         | Required | Description                |
| -------------- | -------- | -------------------------- |
| `Content-Type` | Yes      | Must be `application/json` |
| `x-api-key`    | Yes      | Your API key               |

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
  "explanation": "High spectral artifacts detected consistent with neural TTS."
}
```

### Error Response (4xx/5xx)

```json
{
  "status": "error",
  "message": "Invalid base64 encoding in audioBase64 field."
}
```

### Example curl

```bash
# Encode your MP3 file
AUDIO_BASE64=$(base64 -w 0 your_audio.mp3)

# Make request
curl -X POST http://localhost:8001/api/voice-detection \
  -H "Content-Type: application/json" \
  -H "x-api-key: your-api-key" \
  -d "{
    \"language\": \"English\",
    \"audioFormat\": \"mp3\",
    \"audioBase64\": \"${AUDIO_BASE64}\"
  }"
```

---

## Configuration

All settings are configurable via environment variables:

| Variable                  | Default        | Description                             |
| ------------------------- | -------------- | --------------------------------------- |
| `VOICE_API_KEYS`          | (required)     | Comma-separated valid API keys          |
| `DETECTOR_BACKEND`        | `qc_fallback`  | Detector to use (see below)             |
| `ENABLE_DOCS`             | `0`            | Set to `1` to enable `/docs`            |
| `MAX_MP3_BYTES`           | `15000000`     | Max decoded MP3 size (15 MB)            |
| `MAX_DURATION_SECONDS`    | `300`          | Max audio duration (5 minutes)          |
| `MIN_DURATION_SECONDS`    | `0.5`          | Min audio duration                      |
| `SILENCE_RATIO_THRESHOLD` | `0.80`         | Silence ratio for low-confidence result |
| `PORT`                    | `8080`         | Server port (Cloud Run sets this)       |

---

## Detector Plugin System

The API supports pluggable detection backends via `DETECTOR_BACKEND`:

| Backend       | Status          | Description                                |
| ------------- | --------------- | ------------------------------------------ |
| `qc_fallback` | ✅ Implemented  | QC-based fallback (always available)       |
| `dsp`         | 🔲 Stub         | DSP features (MFCC, spectral analysis)     |
| `ssl`         | 🔲 Stub         | Self-supervised (Wav2Vec2, HuBERT)         |
| `antispoof`   | 🔲 Stub         | Anti-spoofing models (AASIST, RawNet2)     |
| `ensemble`    | 🔲 Stub         | Multi-model ensemble                       |
| `external`    | 🔲 Stub         | External API inference                     |

### Implementing a Detector

1. Create a new file in `detectors/`:

```python
# detectors/my_detector.py
from typing import Dict
import numpy as np
from .base import BaseDetector, PredictionResult

class MyDetector(BaseDetector):
    @property
    def name(self) -> str:
        return "my_detector"
    
    def load(self) -> None:
        # Load model weights here
        self.model = load_my_model("weights.pt")
    
    def predict(
        self,
        language: str,
        mp3_bytes: bytes,
        waveform: np.ndarray,
        sr: int,
        qc: Dict[str, float],
    ) -> PredictionResult:
        # Your detection logic
        score = self.model.predict(waveform)
        return PredictionResult(
            classification="AI_GENERATED" if score > 0.5 else "HUMAN",
            confidenceScore=score,
            explanation="Based on spectral analysis."
        )
```

2. Register in `detectors/registry.py`:

```python
elif backend == "my_detector":
    from .my_detector import MyDetector
    detector = MyDetector()
```

3. Set `DETECTOR_BACKEND=my_detector` in environment.

---

## Docker & Cloud Run Deployment

### Build Docker Image

```bash
docker build -t voice-detection-api .
```

### Run Locally with Docker

```bash
docker run -p 8080:8080 \
  -e VOICE_API_KEYS=your-key \
  -e DETECTOR_BACKEND=qc_fallback \
  voice-detection-api
```

### Deploy to Cloud Run

```bash
# Set your project
export GCP_PROJECT_ID=your-project
export VOICE_API_KEYS=your-production-key

# Run deployment script
chmod +x scripts/deploy_cloudrun.sh
./scripts/deploy_cloudrun.sh
```

Or manually:

```bash
# Build with Cloud Build
gcloud builds submit --tag gcr.io/$GCP_PROJECT_ID/voice-detection-api

# Deploy
gcloud run deploy voice-detection-api \
  --image gcr.io/$GCP_PROJECT_ID/voice-detection-api \
  --region asia-south1 \
  --allow-unauthenticated \
  --set-env-vars="VOICE_API_KEYS=$VOICE_API_KEYS,ENABLE_DOCS=0"
```

**Note**: Cloud Run has ~32 MiB request size limit. Keep MP3 files under `MAX_MP3_BYTES` (15 MB default).

---

## Testing

```bash
# Install test dependencies (already in requirements.txt)
pip install pytest httpx

# Run all tests
pytest tests/ -v

# Run specific test file
pytest tests/test_auth.py -v
```

Tests cover:
- **Authentication**: API key validation
- **Validation**: Request body validation
- **Contract**: Response JSON shape verification

---

## Project Structure

```
├── api.py                 # FastAPI app entrypoint
├── config.py              # Settings (pydantic-settings)
├── base_requests.py       # Request/response Pydantic models
├── platform_services.py   # /api/voice-detection endpoint
├── services/
│   ├── audio_io.py        # FFmpeg MP3 decoder
│   └── qc.py              # Quality control metrics
├── detectors/
│   ├── base.py            # BaseDetector interface
│   ├── registry.py        # Detector initialization
│   ├── qc_fallback.py     # QC-based fallback
│   └── *_stub.py          # Detector stubs
├── tests/
│   ├── conftest.py        # Pytest fixtures
│   ├── test_auth.py
│   ├── test_validation.py
│   └── test_contract.py
├── scripts/
│   ├── run_local.sh       # Local development
│   └── deploy_cloudrun.sh # Cloud Run deployment
├── Dockerfile
├── requirements.txt
└── .env.example
```

---

## License

MIT
