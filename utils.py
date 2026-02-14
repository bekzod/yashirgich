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

# Presidio entity types that take priority over rubai
PRESIDIO_PRIORITY_TYPES = {"url", "phone", "email"}


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


def detect_with_presidio(text: str, presidio_analyzer) -> list[dict]:
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


def detect_with_rubai(text: str, ner_pipeline, original_text: str | None = None, position_map: list[int] | None = None) -> list[dict]:
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
