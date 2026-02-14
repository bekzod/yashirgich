# PII Detection API

A FastAPI-based service for detecting and masking Personally Identifiable Information (PII) in text, optimized for Uzbek language with support for both Latin and Cyrillic scripts.

## Features

- **Dual Detection System**
  - **Primary**: rubai model (`islomov/rubai-PII-detection-v1-latin`) - specialized for Uzbek text
  - **Backup**: Microsoft Presidio - rule-based pattern matching for standard formats

- **Multi-Script Support**
  - Latin script (default)
  - Cyrillic script (automatic transliteration)
  - Multi-language support (English, Russian, Uzbek)

- **Comprehensive PII Detection**
  - Names (NAME)
  - Phone numbers (PHONE)
  - Email addresses (EMAIL)
  - Physical addresses (ADDRESS)
  - Dates (DATE)
  - Document IDs (DOCUMENT_ID)
  - Credit card numbers (CARD_NUMBER)
  - IP addresses (IP_ADDRESS)
  - URLs (URL)
  - IBAN codes (IBAN)
  - SSN and other identifiers

## Installation

### Prerequisites

- Python 3.8+
- pip

### Setup

1. Clone the repository:
```bash
git clone <repository-url>
cd yashirgich
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Download required spaCy models:
```bash
python -m spacy download en_core_web_sm
python -m spacy download ru_core_news_sm
```

## Usage

### Starting the Server

```bash
uvicorn main:app --reload
```

The server will start at `http://localhost:8000`

### API Documentation

Interactive API documentation is available at:
- Swagger UI: `http://localhost:8000/docs`
- ReDoc: `http://localhost:8000/redoc`

### Web Interface

Access the web UI at `http://localhost:8000` for an interactive interface to test PII detection.

## API Endpoints

### 1. Detect PII in Text

**POST** `/detect`

Detect and mask PII in provided text.

**Request:**
```json
{
  "text": "Sardor Rustamov telefon raqami 90 123 45 67, email: test@example.com"
}
```

**Response:**
```json
{
  "masked_text": "<<NAME_1>> telefon raqami <<PHONE_1>>, email: <<EMAIL_1>>",
  "entities": [
    {
      "original": "Sardor Rustamov",
      "masked_as": "<<NAME_1>>",
      "entity_type": "NAME",
      "start": 0,
      "end": 15,
      "source": "rubai"
    },
    {
      "original": "90 123 45 67",
      "masked_as": "<<PHONE_1>>",
      "entity_type": "PHONE",
      "start": 32,
      "end": 44,
      "source": "rubai"
    },
    {
      "original": "test@example.com",
      "masked_as": "<<EMAIL_1>>",
      "entity_type": "EMAIL",
      "start": 53,
      "end": 69,
      "source": "presidio"
    }
  ]
}
```

### 2. Upload File

**POST** `/upload`

Upload a text file and detect PII in its contents.

**Requirements:**
- File format: Plain text (.txt, .md, etc.)
- Encoding: UTF-8
- Content: Non-empty

**Response:** Same format as `/detect` endpoint

### 3. Health Check

**GET** `/health`

Check service health and model status.

**Response:**
```json
{
  "status": "healthy",
  "models": {
    "primary": "islomov/rubai-PII-detection-v1-latin",
    "backup": "Microsoft Presidio"
  }
}
```

## Examples

### Python Client

```python
import requests

# Detect PII
response = requests.post(
    "http://localhost:8000/detect",
    json={"text": "Sardor Rustamov telefon raqami 90 123 45 67"}
)
result = response.json()
print(result["masked_text"])
print(result["entities"])
```

### cURL

```bash
# Detect PII
curl -X POST "http://localhost:8000/detect" \
  -H "Content-Type: application/json" \
  -d '{"text": "Sardor Rustamov telefon raqami 90 123 45 67"}'

# Upload file
curl -X POST "http://localhost:8000/upload" \
  -F "file=@document.txt"

# Health check
curl "http://localhost:8000/health"
```

### JavaScript/Node.js

```javascript
// Detect PII
const response = await fetch('http://localhost:8000/detect', {
  method: 'POST',
  headers: { 'Content-Type': 'application/json' },
  body: JSON.stringify({
    text: 'Sardor Rustamov telefon raqami 90 123 45 67'
  })
});
const result = await response.json();
console.log(result.masked_text);
```

## How It Works

### Detection Process

1. **Text Analysis**
   - Checks if text contains Cyrillic characters
   - Automatically transliterates Cyrillic to Latin for rubai model

2. **Dual Detection**
   - rubai model analyzes text for Uzbek-specific PII patterns
   - Presidio runs pattern matching for standard formats (email, URL, etc.)

3. **Entity Merging**
   - rubai detections take priority
   - Presidio fills gaps for undetected regions
   - Eliminates overlapping detections

4. **Masking**
   - Each entity type gets unique sequential labels
   - Format: `<<TYPE_NUMBER>>` (e.g., `<<NAME_1>>`, `<<PHONE_2>>`)
   - Original positions preserved in response

### Cyrillic Support

The service automatically handles Cyrillic text:

```python
# Input (Cyrillic)
"Сардор Рустамов телефон рақами 90 123 45 67"

# Automatic transliteration for rubai model
# Detection and masking work seamlessly

# Output
"<<NAME_1>> телефон рақами <<PHONE_1>>"
```

## Configuration

### Environment Variables

You can customize the service using environment variables:

```bash
# Server configuration
export HOST=0.0.0.0
export PORT=8000

# Model configuration (optional)
export RUBAI_MODEL=islomov/rubai-PII-detection-v1-latin
```

### Production Deployment

For production deployment:

```bash
# Using Gunicorn with Uvicorn workers
gunicorn main:app \
  --workers 4 \
  --worker-class uvicorn.workers.UvicornWorker \
  --bind 0.0.0.0:8000
```

## Dependencies

Main dependencies (see `requirements.txt` for full list):

- **FastAPI**: Web framework
- **transformers**: HuggingFace models (rubai)
- **presidio-analyzer**: Microsoft Presidio
- **spacy**: NLP processing
- **torch**: Deep learning backend
- **uvicorn**: ASGI server

## Performance Considerations

- **First Request**: Initial request may be slower due to model loading
- **Subsequent Requests**: Fast inference after models are loaded
- **Memory Usage**: Approximately 1-2GB RAM for loaded models
- **Concurrency**: Supports concurrent requests (use multiple workers in production)

## Error Handling

The API returns standard HTTP status codes:

- **200**: Success
- **400**: Bad request (empty text, invalid file)
- **503**: Service unavailable (models not loaded)
- **500**: Internal server error

Example error response:
```json
{
  "detail": "Text cannot be empty"
}
```

## Limitations

- Supports primarily Uzbek, English, and Russian text
- Best performance on Latin and Cyrillic scripts
- File uploads limited to UTF-8 encoded text
- Some uncommon PII patterns may not be detected

## Contributing

Contributions are welcome. Please ensure:
- Code follows existing style conventions
- Tests pass before submitting
- Documentation is updated for new features

## License

[Specify your license here]

## Support

For issues or questions:
- Open an issue in the repository
- Check API documentation at `/docs`
- Review examples in this README

## Acknowledgments

- **rubai model**: [islomov/rubai-PII-detection-v1-latin](https://huggingface.co/islomov/rubai-PII-detection-v1-latin)
- **Microsoft Presidio**: [presidio-analyzer](https://github.com/microsoft/presidio)
- **spaCy**: [spacy.io](https://spacy.io)
