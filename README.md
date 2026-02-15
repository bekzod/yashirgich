# PII Detection & OpenAI Proxy API

A FastAPI-based service for detecting and masking Personally Identifiable Information (PII) in text, optimized for Uzbek language with support for both Latin and Cyrillic scripts. Includes an OpenAI-compatible proxy that automatically masks PII in requests and restores it in responses.

## Features

- **Dual Detection System**
  - **Primary**: rubai model by **[Sardor Islomov](https://huggingface.co/islomov)** ([`rubai-PII-detection-v1-latin`](https://huggingface.co/islomov/rubai-PII-detection-v1-latin)) - specialized for Uzbek text
  - **Backup**: Microsoft Presidio - rule-based pattern matching for standard formats

- **OpenAI Proxy with PII Protection**
  - Drop-in replacement for OpenAI API endpoints
  - Automatic PII masking in outgoing requests
  - Automatic PII restoration in responses
  - Supports streaming and non-streaming requests
  - Redis-backed or in-memory storage for replacement mappings

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

4. Configure environment variables (optional):
```bash
cp .env.example .env
# Edit .env with your settings:
# - OPENAI_API_KEY: Your OpenAI API key (for proxy functionality)
# - REDIS_URL: Redis connection URL (optional, defaults to in-memory storage)
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

### 4. OpenAI Proxy with PII Protection

**POST** `/proxy/v1/chat/completions`

OpenAI-compatible chat completions endpoint that automatically masks PII in your requests before sending to OpenAI and restores it in responses.

**Features:**
- Transparent PII masking/restoration
- Supports streaming and non-streaming
- Drop-in replacement for OpenAI API
- Redis or in-memory storage for mappings

**Setup:**
1. Set your OpenAI API key:
```bash
export OPENAI_API_KEY=sk-...
# Or pass in Authorization header
```

2. Optional: Configure Redis for distributed deployments:
```bash
export REDIS_URL=redis://localhost:6379/0
```

**Usage with OpenAI SDK:**

```python
from openai import OpenAI

# Point to the proxy instead of OpenAI directly
client = OpenAI(
    base_url="http://localhost:8000/proxy/v1",
    api_key="sk-..."  # Your OpenAI API key
)

# Use normally - PII is automatically masked/restored
response = client.chat.completions.create(
    model="gpt-4",
    messages=[{
        "role": "user",
        "content": "Sardor Rustamov telefon raqami 90 123 45 67"
    }]
)
```

**What Happens:**
1. Your request: `"Sardor Rustamov telefon raqami 90 123 45 67"`
2. Masked request to OpenAI: `"<<NAME_1>> telefon raqami <<PHONE_1>>"`
3. OpenAI processes masked data
4. Response is automatically restored with original PII
5. You receive the complete response with real names/numbers

**Streaming Example:**

```python
stream = client.chat.completions.create(
    model="gpt-4",
    messages=[{"role": "user", "content": "Call Sardor at 90 123 45 67"}],
    stream=True
)

for chunk in stream:
    if chunk.choices[0].delta.content:
        print(chunk.choices[0].delta.content, end="")
```

**cURL Example:**

```bash
curl -X POST "http://localhost:8000/proxy/v1/chat/completions" \
  -H "Authorization: Bearer sk-..." \
  -H "Content-Type: application/json" \
  -d '{
    "model": "gpt-4",
    "messages": [
      {"role": "user", "content": "Sardor Rustamov telefon raqami 90 123 45 67"}
    ]
  }'
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

# OpenAI Proxy configuration
export OPENAI_API_KEY=sk-...  # Required for proxy functionality
export REDIS_URL=redis://localhost:6379/0  # Optional, for distributed deployments

# Model configuration (optional)
export RUBAI_MODEL=islomov/rubai-PII-detection-v1-latin
```

### Redis Setup (Optional)

For production deployments or distributed systems, use Redis to store PII replacement mappings:

```bash
# Install Redis
brew install redis  # macOS
# or
apt-get install redis  # Ubuntu

# Start Redis
redis-server

# Install Python Redis client (already in requirements.txt)
pip install redis
```

Without Redis, the service uses in-memory storage (suitable for single-instance deployments).

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
- **httpx**: HTTP client for OpenAI proxy
- **redis** (optional): For distributed PII mapping storage

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

## Use Cases

### 1. Direct PII Detection
Use the `/detect` endpoint for standalone PII detection and masking in your applications.

### 2. OpenAI Proxy for Privacy
Route your OpenAI API calls through the proxy to ensure PII never leaves your infrastructure in plaintext:
- Customer support chatbots
- Document analysis systems
- Data processing pipelines
- Any LLM application handling sensitive information

### 3. Compliance and Data Protection
- GDPR compliance for EU users
- HIPAA compliance for healthcare data
- General data protection best practices

## Limitations

- Supports primarily Uzbek, English, and Russian text
- Best performance on Latin and Cyrillic scripts
- File uploads limited to UTF-8 encoded text
- Some uncommon PII patterns may not be detected
- OpenAI proxy adds slight latency for PII detection/restoration

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
