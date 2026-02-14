from pathlib import Path
from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
from transformers import pipeline
from presidio_analyzer import AnalyzerEngine, RecognizerRegistry
from presidio_analyzer.nlp_engine import NlpEngineProvider
from contextlib import asynccontextmanager

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
        aggregation_strategy="simple"
    )

    # Load Presidio analyzer for pattern-based detection (emails, phones, IPs, etc.)
    provider = NlpEngineProvider(nlp_configuration={
        "nlp_engine_name": "spacy",
        "models": [
            {"lang_code": "en", "model_name": "en_core_web_sm"},
            {"lang_code": "ru", "model_name": "ru_core_news_sm"}
        ]
    })
    nlp_engine = provider.create_engine()
    registry = RecognizerRegistry()
    registry.load_predefined_recognizers(nlp_engine=nlp_engine, languages=["en", "ru"])
    presidio_analyzer = AnalyzerEngine(nlp_engine=nlp_engine, registry=registry)

    yield


app = FastAPI(
    title="PII Detection API",
    description="Detect and mask PII using rubai model (Uzbek) and Microsoft Presidio (patterns). Supports Latin/Cyrillic.",
    version="1.0.0",
    lifespan=lifespan
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


# Uzbek Cyrillic to Latin transliteration map
CYRILLIC_TO_LATIN = {
    'а': 'a', 'б': 'b', 'в': 'v', 'г': 'g', 'д': 'd', 'е': 'e', 'ё': 'yo',
    'ж': 'j', 'з': 'z', 'и': 'i', 'й': 'y', 'к': 'k', 'л': 'l', 'м': 'm',
    'н': 'n', 'о': 'o', 'п': 'p', 'р': 'r', 'с': 's', 'т': 't', 'у': 'u',
    'ф': 'f', 'х': 'x', 'ц': 'ts', 'ч': 'ch', 'ш': 'sh', 'щ': 'sh',
    'ъ': '', 'ы': 'y', 'ь': '', 'э': 'e', 'ю': 'yu', 'я': 'ya',
    'А': 'A', 'Б': 'B', 'В': 'V', 'Г': 'G', 'Д': 'D', 'Е': 'E', 'Ё': 'Yo',
    'Ж': 'J', 'З': 'Z', 'И': 'I', 'Й': 'Y', 'К': 'K', 'Л': 'L', 'М': 'M',
    'Н': 'N', 'О': 'O', 'П': 'P', 'Р': 'R', 'С': 'S', 'Т': 'T', 'У': 'U',
    'Ф': 'F', 'Х': 'X', 'Ц': 'Ts', 'Ч': 'Ch', 'Ш': 'Sh', 'Щ': 'Sh',
    'Ъ': '', 'Ы': 'Y', 'Ь': '', 'Э': 'E', 'Ю': 'Yu', 'Я': 'Ya',
    # Uzbek specific characters
    'ў': "o'", 'ҳ': 'h', 'қ': 'q', 'ғ': "g'",
    'Ў': "O'", 'Ҳ': 'H', 'Қ': 'Q', 'Ғ': "G'",
}


def transliterate_uzbek_cyrillic(text: str) -> tuple[str, list[int]]:
    """
    Transliterate Uzbek Cyrillic to Latin.
    Returns the transliterated text and a mapping from latin positions to original positions.
    """
    result = []
    # Maps each position in the result to the original position
    position_map = []

    for orig_idx, char in enumerate(text):
        latin = CYRILLIC_TO_LATIN.get(char, char)
        for c in latin:
            result.append(c)
            position_map.append(orig_idx)

    # Add end position for boundary calculations
    position_map.append(len(text))

    return ''.join(result), position_map


def map_positions_back(start: int, end: int, position_map: list[int], original_text: str) -> tuple[int, int]:
    """Map positions from transliterated text back to original text."""
    if start >= len(position_map):
        start = len(position_map) - 1
    if end > len(position_map):
        end = len(position_map)

    orig_start = position_map[start]
    # For end, we need the position after the last character
    orig_end = position_map[end - 1] + 1 if end > 0 else 0

    return orig_start, orig_end


def has_cyrillic(text: str) -> bool:
    """Check if text contains Cyrillic characters."""
    return any('\u0400' <= c <= '\u04FF' for c in text)


# Presidio entity types to detect (limited set for pattern-based detection)
PRESIDIO_ENTITIES = {
    "EMAIL_ADDRESS",
    "PHONE_NUMBER",
    "CREDIT_CARD",
    "IP_ADDRESS",
    "URL",
    "IBAN_CODE",
    "US_SSN",
    "DATE_TIME",
}

# Map Presidio entity types to our naming convention
PRESIDIO_TYPE_MAP = {
    "EMAIL_ADDRESS": "email",
    "PHONE_NUMBER": "phone",
    "CREDIT_CARD": "card_number",
    "IP_ADDRESS": "ip_address",
    "URL": "url",
    "IBAN_CODE": "iban",
    "US_SSN": "ssn",
    "DATE_TIME": "date",
}


def detect_with_presidio(text: str) -> list[dict]:
    """Detect PII using Presidio for specific pattern-based entities."""
    entities = []
    seen_spans: set[tuple[int, int]] = set()

    for lang in ["en", "ru"]:
        results = presidio_analyzer.analyze(
            text=text,
            language=lang,
            entities=list(PRESIDIO_ENTITIES)
        )
        for r in results:
            # Skip if we already detected this span
            if (r.start, r.end) in seen_spans:
                continue
            seen_spans.add((r.start, r.end))

            entity_type = PRESIDIO_TYPE_MAP.get(r.entity_type, r.entity_type.lower())
            entities.append({
                "start": r.start,
                "end": r.end,
                "text": text[r.start:r.end],
                "type": entity_type,
                "source": "presidio"
            })

    return entities


def detect_with_rubai(text: str, original_text: str | None = None, position_map: list[int] | None = None) -> list[dict]:
    """
    Detect PII using the rubai model.
    If position_map is provided, maps positions back to original text.
    """
    results = ner_pipeline(text)
    entities = []
    for r in results:
        # Skip TEXT and DATE entities (DATE handled by Presidio to avoid IP conflicts)
        if r["entity_group"] in ("TEXT", "DATE"):
            continue

        start, end = r["start"], r["end"]

        # Map positions back if we transliterated
        if position_map is not None and original_text is not None:
            orig_start, orig_end = map_positions_back(start, end, position_map, original_text)
            entity_text = original_text[orig_start:orig_end]
            start, end = orig_start, orig_end
        else:
            entity_text = r["word"]

        entities.append({
            "start": start,
            "end": end,
            "text": entity_text,
            "type": r["entity_group"].lower(),
            "source": "rubai"
        })
    return entities


# Presidio entity types that take priority over rubai
PRESIDIO_PRIORITY_TYPES = {"url", "phone", "email"}


def dedupe_presidio_entities(entities: list[dict]) -> list[dict]:
    """
    Remove overlapping Presidio entities, prioritizing email over URL.
    """
    # Sort by type priority (email first) then by span length (longer first)
    type_priority = {"email": 0, "phone": 1, "url": 2}
    sorted_entities = sorted(
        entities,
        key=lambda e: (type_priority.get(e["type"], 99), -(e["end"] - e["start"]))
    )

    result = []
    for entity in sorted_entities:
        overlaps = False
        for existing in result:
            if not (entity["end"] <= existing["start"] or entity["start"] >= existing["end"]):
                overlaps = True
                break
        if not overlaps:
            result.append(entity)

    return result


def merge_entities(rubai_entities: list[dict], presidio_entities: list[dict]) -> list[dict]:
    """
    Merge entities from both sources.
    Presidio EMAIL/URL/PHONE take priority; rubai fills gaps for other types.
    """
    # Dedupe presidio entities (email takes priority over URL)
    presidio_entities = dedupe_presidio_entities(presidio_entities)

    # Separate priority presidio entities (EMAIL, URL, PHONE)
    priority_entities = [e for e in presidio_entities if e["type"] in PRESIDIO_PRIORITY_TYPES]
    other_presidio = [e for e in presidio_entities if e["type"] not in PRESIDIO_PRIORITY_TYPES]

    # Start with priority presidio entities
    merged = priority_entities.copy()

    # Add rubai entities that don't overlap with priority entities
    for r_entity in rubai_entities:
        overlaps = False
        for p_entity in priority_entities:
            if not (r_entity["end"] <= p_entity["start"] or r_entity["start"] >= p_entity["end"]):
                overlaps = True
                break
        if not overlaps:
            merged.append(r_entity)

    # Add other presidio entities that don't overlap with what we have
    for p_entity in other_presidio:
        overlaps = False
        for existing in merged:
            if not (p_entity["end"] <= existing["start"] or p_entity["start"] >= existing["end"]):
                overlaps = True
                break
        if not overlaps:
            merged.append(p_entity)

    # Sort by start position
    merged.sort(key=lambda x: x["start"])
    return merged


def mask_pii(text: str) -> PIIResponse:
    """Detect PII using rubai model and Presidio, return masked text with entity information."""
    # Detect with Presidio first (pattern-based: emails, phones, IPs, etc.)
    presidio_entities = detect_with_presidio(text)

    # Check if text contains Cyrillic - if so, transliterate for rubai model
    if has_cyrillic(text):
        transliterated, position_map = transliterate_uzbek_cyrillic(text)
        rubai_entities = detect_with_rubai(transliterated, text, position_map)
    else:
        rubai_entities = detect_with_rubai(text)

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
        entity["mask_label"] = f"<<{entity_type.upper()}_{entity_counters[entity_type]}>>"

    # Replace text (in reverse order to preserve positions)
    masked_text = text
    entities: list[MaskedEntity] = []

    for entity in reversed(all_entities):
        start, end = entity["start"], entity["end"]
        mask_label = entity["mask_label"]

        entities.append(MaskedEntity(
            original=entity["text"],
            masked_as=mask_label,
            entity_type=entity["type"].upper(),
            start=start,
            end=end,
            source=entity["source"]
        ))

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
            "backup": "Microsoft Presidio"
        }
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
