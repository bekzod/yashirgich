import json
import os
import uuid
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Protocol

import httpx
from dotenv import load_dotenv
from fastapi import FastAPI, File, Header, HTTPException, Request, UploadFile
from fastapi.responses import HTMLResponse, StreamingResponse
from presidio_analyzer import AnalyzerEngine, RecognizerRegistry
from presidio_analyzer.nlp_engine import NlpEngineProvider
from pydantic import BaseModel
from transformers import pipeline

from utils import (
    detect_with_presidio,
    detect_with_rubai,
    has_cyrillic,
    merge_entities,
    transliterate_uzbek_cyrillic,
)

# Load environment variables from .env file
load_dotenv()

# Global instances
ner_pipeline = None
presidio_analyzer = None
replacement_store: "ReplacementStore | None" = None


# --- Replacement Store ---


class ReplacementStore(Protocol):
    """Protocol for storing PII replacement mappings."""

    async def save(
        self, request_id: str, replacement_map: dict[str, str], ttl: int = 3600
    ) -> None: ...

    async def get(self, request_id: str) -> dict[str, str] | None: ...

    async def delete(self, request_id: str) -> None: ...


class InMemoryReplacementStore:
    """In-memory store for PII replacement mappings with TTL support."""

    def __init__(self):
        self._store: dict[str, tuple[dict[str, str], float]] = {}

    async def save(
        self, request_id: str, replacement_map: dict[str, str], ttl: int = 3600
    ) -> None:
        import time

        expiry = time.time() + ttl
        self._store[request_id] = (replacement_map, expiry)

    async def get(self, request_id: str) -> dict[str, str] | None:
        import time

        if request_id not in self._store:
            return None
        data, expiry = self._store[request_id]
        if time.time() > expiry:
            del self._store[request_id]
            return None
        return data

    async def delete(self, request_id: str) -> None:
        self._store.pop(request_id, None)


class RedisReplacementStore:
    """Redis-backed store for PII replacement mappings."""

    def __init__(self, redis_client):
        self._redis = redis_client
        self._prefix = "pii:replacement:"

    async def save(
        self, request_id: str, replacement_map: dict[str, str], ttl: int = 3600
    ) -> None:
        key = f"{self._prefix}{request_id}"
        await self._redis.setex(key, ttl, json.dumps(replacement_map))

    async def get(self, request_id: str) -> dict[str, str] | None:
        key = f"{self._prefix}{request_id}"
        data = await self._redis.get(key)
        if data is None:
            return None
        return json.loads(data)

    async def delete(self, request_id: str) -> None:
        key = f"{self._prefix}{request_id}"
        await self._redis.delete(key)


async def init_replacement_store() -> ReplacementStore:
    """Initialize replacement store - Redis if available, else in-memory."""
    redis_url = os.getenv("REDIS_URL")
    if redis_url:
        try:
            import redis.asyncio as aioredis

            client = aioredis.from_url(redis_url)
            await client.ping()
            print("Using Redis for replacement storage")
            return RedisReplacementStore(client)
        except ImportError:
            print("redis package not installed, using in-memory storage")
        except Exception as e:
            print(f"Redis unavailable ({e}), falling back to in-memory storage")

    print("Using in-memory replacement storage")
    return InMemoryReplacementStore()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load models on startup."""
    global ner_pipeline, presidio_analyzer, replacement_store

    # Initialize replacement store (Redis if available, else in-memory)
    replacement_store = await init_replacement_store()

    # Load rubai model (optimized for Uzbek)
    ner_pipeline = pipeline(
        "ner",
        model="islomov/rubai-PII-detection-v1-latin",
        aggregation_strategy="simple",
    )

    # Load Presidio analyzer for pattern-based detection (emails, phones, IPs, etc.)
    provider = NlpEngineProvider(
        nlp_configuration={
            "nlp_engine_name": "spacy",
            "models": [
                {"lang_code": "en", "model_name": "en_core_web_sm"},
                {"lang_code": "ru", "model_name": "ru_core_news_sm"},
            ],
        }
    )
    nlp_engine = provider.create_engine()
    registry = RecognizerRegistry()
    registry.load_predefined_recognizers(nlp_engine=nlp_engine, languages=["en", "ru"])
    presidio_analyzer = AnalyzerEngine(nlp_engine=nlp_engine, registry=registry)

    yield


app = FastAPI(
    title="PII Detection & OpenAI Proxy API",
    description="Detect and mask PII using rubai model (Uzbek) and Microsoft Presidio (patterns). Includes OpenAI proxy with automatic PII masking/restoration. Supports Latin/Cyrillic.",
    version="1.0.0",
    lifespan=lifespan,
)


class PIIRequest(BaseModel):
    text: str

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "text": "Sardor Rustamov telefon raqami 90 123 45 67, email: test@example.com"
                }
            ]
        }
    }


class MaskedEntity(BaseModel):
    original: str
    masked_as: str
    entity_type: str
    start: int
    end: int
    source: str  # "rubai" or "presidio"


class PIIResponse(BaseModel):
    masked_text: str
    entities: list[MaskedEntity]


def mask_pii(text: str) -> PIIResponse:
    """Detect PII using rubai model and Presidio, return masked text with entity information."""
    # Detect with Presidio first (pattern-based: emails, phones, IPs, etc.)
    presidio_entities = detect_with_presidio(text, presidio_analyzer)

    # Check if text contains Cyrillic - if so, transliterate for rubai model
    if has_cyrillic(text):
        transliterated, position_map = transliterate_uzbek_cyrillic(text)
        rubai_entities = detect_with_rubai(
            transliterated, ner_pipeline, text, position_map
        )
    else:
        rubai_entities = detect_with_rubai(text, ner_pipeline)

    # Merge entities (rubai takes priority, presidio fills gaps)
    all_entities = merge_entities(rubai_entities, presidio_entities)

    # Filter out ORGANIZATION entities (we don't want to mask organization names)
    all_entities = [e for e in all_entities if e["type"].lower() != "organization"]

    # Track counters for each entity type
    entity_counters: dict[str, int] = {}

    # Assign mask labels with unique format to avoid conflicts
    for entity in all_entities:
        entity_type = entity["type"]
        entity_counters[entity_type] = entity_counters.get(entity_type, 0) + 1
        entity["mask_label"] = (
            f"<<{entity_type.upper()}_{entity_counters[entity_type]}>>"
        )

    # Replace text (in reverse order to preserve positions)
    masked_text = text
    entities: list[MaskedEntity] = []

    for entity in reversed(all_entities):
        start, end = entity["start"], entity["end"]
        mask_label = entity["mask_label"]

        entities.append(
            MaskedEntity(
                original=entity["text"],
                masked_as=mask_label,
                entity_type=entity["type"].upper(),
                start=start,
                end=end,
                source=entity["source"],
            )
        )

        masked_text = masked_text[:start] + mask_label + masked_text[end:]

    # Reverse to have entities in original text order
    entities.reverse()

    return PIIResponse(masked_text=masked_text, entities=entities)


@app.post("/detect", response_model=PIIResponse, tags=["PII Detection"])
async def detect_pii(request: PIIRequest):
    """Detect and mask PII using rubai + Presidio. Supports Latin/Cyrillic."""
    if not request.text.strip():
        raise HTTPException(status_code=400, detail="Text cannot be empty")

    return mask_pii(request.text)


@app.get("/health", tags=["System"])
async def health_check():
    """Check if service and models are loaded."""
    if ner_pipeline is None or presidio_analyzer is None:
        raise HTTPException(status_code=503, detail="Models not loaded")
    return {
        "status": "healthy",
        "models": {
            "primary": "islomov/rubai-PII-detection-v1-latin",
            "backup": "Microsoft Presidio",
        },
    }


@app.post("/upload", response_model=PIIResponse, tags=["PII Detection"])
async def upload_file(file: UploadFile = File(...)):
    """Upload a UTF-8 text file and detect PII."""
    content = await file.read()
    try:
        text = content.decode("utf-8")
    except UnicodeDecodeError:
        raise HTTPException(status_code=400, detail="File must be UTF-8 encoded text")

    if not text.strip():
        raise HTTPException(status_code=400, detail="File is empty")

    return mask_pii(text)


@app.get("/", response_class=HTMLResponse, tags=["UI"])
async def ui():
    """Web UI for PII detection."""
    template_path = Path(__file__).parent / "templates" / "index.html"
    return template_path.read_text()


# --- OpenAI Proxy Endpoints ---

OPENAI_API_BASE = "https://api.openai.com/v1"


def mask_text_simple(text: str) -> tuple[str, dict[str, str]]:
    """Mask PII in text and return masked text with replacement map."""
    result = mask_pii(text)
    replacement_map = {e.masked_as: e.original for e in result.entities}
    return result.masked_text, replacement_map


def restore_pii(text: str, replacement_map: dict[str, str]) -> str:
    """Restore original PII values in text using the replacement map."""
    restored = text
    for masked, original in replacement_map.items():
        restored = restored.replace(masked, original)
    return restored


def process_messages_for_pii(
    messages: list[dict],
) -> tuple[list[dict], dict[str, str]]:
    """Process OpenAI messages, mask PII in content fields."""
    combined_map: dict[str, str] = {}
    processed_messages = []

    for msg in messages:
        new_msg = msg.copy()
        content = msg.get("content")

        if isinstance(content, str):
            masked_content, rep_map = mask_text_simple(content)
            new_msg["content"] = masked_content
            combined_map.update(rep_map)
        elif isinstance(content, list):
            # Handle multimodal content (text + images)
            new_content = []
            for part in content:
                if isinstance(part, dict) and part.get("type") == "text":
                    text_val = part.get("text", "")
                    masked_text, rep_map = mask_text_simple(text_val)
                    new_content.append({**part, "text": masked_text})
                    combined_map.update(rep_map)
                else:
                    new_content.append(part)
            new_msg["content"] = new_content

        processed_messages.append(new_msg)

    return processed_messages, combined_map


@app.post("/proxy/v1/chat/completions", tags=["OpenAI Proxy"])
async def openai_chat_completions_proxy(
    request: Request,
    authorization: str | None = Header(default=None),
):
    """Proxy for OpenAI chat completions with PII masking.

    - Masks PII in request messages before sending to OpenAI
    - Restores original PII in the response
    - Supports both streaming and non-streaming requests
    - Stores replacement mappings in Redis (if available) or in-memory
    """
    # Get API key from header or environment
    api_key = None
    if authorization and authorization.startswith("Bearer "):
        api_key = authorization[7:]
    if not api_key:
        api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise HTTPException(
            status_code=401,
            detail="OpenAI API key required. Set OPENAI_API_KEY or pass Authorization header.",
        )

    # Parse request body
    try:
        body = await request.json()
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid JSON body")

    messages = body.get("messages", [])
    if not messages:
        raise HTTPException(status_code=400, detail="Messages required")

    # Mask PII in messages
    processed_messages, replacement_map = process_messages_for_pii(messages)
    body["messages"] = processed_messages

    # Store replacement map with unique request ID
    request_id = str(uuid.uuid4())
    await replacement_store.save(request_id, replacement_map)

    is_streaming = body.get("stream", False)

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }

    if is_streaming:
        return await _handle_streaming_request(body, headers, request_id)
    else:
        return await _handle_non_streaming_request(body, headers, request_id)


async def _handle_non_streaming_request(
    body: dict, headers: dict, request_id: str
) -> dict:
    """Handle non-streaming OpenAI request."""
    try:
        async with httpx.AsyncClient(timeout=120.0) as client:
            response = await client.post(
                f"{OPENAI_API_BASE}/chat/completions",
                json=body,
                headers=headers,
            )

        if response.status_code != 200:
            raise HTTPException(status_code=response.status_code, detail=response.text)

        result = response.json()

        # Get replacement map from store
        replacement_map = await replacement_store.get(request_id)
        if replacement_map:
            # Restore PII in response choices
            if "choices" in result:
                for choice in result["choices"]:
                    if "message" in choice and "content" in choice["message"]:
                        content = choice["message"]["content"]
                        if content:
                            choice["message"]["content"] = restore_pii(
                                content, replacement_map
                            )

        return result
    finally:
        # Clean up stored replacement map
        await replacement_store.delete(request_id)


async def _handle_streaming_request(body: dict, headers: dict, request_id: str):
    """Handle streaming OpenAI request."""

    async def stream_generator():
        try:
            # Get replacement map from store
            replacement_map = await replacement_store.get(request_id) or {}

            async with httpx.AsyncClient(timeout=120.0) as client:
                async with client.stream(
                    "POST",
                    f"{OPENAI_API_BASE}/chat/completions",
                    json=body,
                    headers=headers,
                ) as response:
                    if response.status_code != 200:
                        error_body = await response.aread()
                        yield f"data: {json.dumps({'error': error_body.decode()})}\n\n"
                        return

                    async for line in response.aiter_lines():
                        if not line:
                            continue
                        if line.startswith("data: "):
                            data_str = line[6:]
                            if data_str.strip() == "[DONE]":
                                yield "data: [DONE]\n\n"
                                continue
                            try:
                                data = json.loads(data_str)
                                # Restore PII in streamed content
                                if "choices" in data:
                                    for choice in data["choices"]:
                                        delta = choice.get("delta", {})
                                        if "content" in delta and delta["content"]:
                                            delta["content"] = restore_pii(
                                                delta["content"], replacement_map
                                            )
                                yield f"data: {json.dumps(data)}\n\n"
                            except json.JSONDecodeError:
                                yield f"{line}\n"
                        else:
                            yield f"{line}\n"
        finally:
            # Clean up stored replacement map
            await replacement_store.delete(request_id)

    return StreamingResponse(
        stream_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
        },
    )
