from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import HTMLResponse
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

# Global instances
ner_pipeline = None
presidio_analyzer = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load models on startup."""
    global ner_pipeline, presidio_analyzer

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
    title="PII Detection API",
    description="Detect and mask PII using rubai model (Uzbek) and Microsoft Presidio (patterns). Supports Latin/Cyrillic.",
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
