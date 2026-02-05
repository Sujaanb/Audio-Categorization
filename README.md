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

### Error Response (4xx/5xx)

```json
{
  "status": "error",
  "message": "Invalid base64 encoding in audioBase64 field."
}
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
│   └── qc.py              # Quality control metrics
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
