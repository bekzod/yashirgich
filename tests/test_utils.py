"""Tests for utils.py - PII detection utilities."""

import pytest

from src.utils import (
    CYRILLIC_TO_LATIN,
    PRESIDIO_ENTITIES,
    PRESIDIO_PRIORITY_TYPES,
    PRESIDIO_TYPE_MAP,
    dedupe_presidio_entities,
    has_cyrillic,
    map_positions_back,
    merge_entities,
    transliterate_uzbek_cyrillic,
)


class TestTransliterateUzbekCyrillic:
    """Tests for transliterate_uzbek_cyrillic function."""

    def test_basic_cyrillic_to_latin(self):
        """Test basic Cyrillic to Latin transliteration."""
        text = "абв"
        result, position_map = transliterate_uzbek_cyrillic(text)
        assert result == "abv"
        assert position_map == [0, 1, 2, 3]

    def test_uzbek_specific_characters(self):
        """Test Uzbek-specific characters."""
        text = "ўҳқғ"
        result, position_map = transliterate_uzbek_cyrillic(text)
        assert result == "o'hqg'"

    def test_multi_char_mappings(self):
        """Test characters that map to multiple Latin characters."""
        text = "ёчш"
        result, position_map = transliterate_uzbek_cyrillic(text)
        assert result == "yochsh"
        # ё->yo (2 chars), ч->ch (2 chars), ш->sh (2 chars)
        assert position_map == [0, 0, 1, 1, 2, 2, 3]

    def test_mixed_text(self):
        """Test mixed Cyrillic and Latin text."""
        text = "Hello Мир"
        result, position_map = transliterate_uzbek_cyrillic(text)
        assert result == "Hello Mir"

    def test_empty_string(self):
        """Test empty string."""
        result, position_map = transliterate_uzbek_cyrillic("")
        assert result == ""
        assert position_map == [0]

    def test_uppercase_cyrillic(self):
        """Test uppercase Cyrillic characters."""
        text = "АБВ"
        result, position_map = transliterate_uzbek_cyrillic(text)
        assert result == "ABV"

    def test_silent_characters(self):
        """Test characters that map to empty strings."""
        text = "объект"
        result, _ = transliterate_uzbek_cyrillic(text)
        # ъ and ь map to empty strings
        assert "ъ" not in result
        assert "ь" not in result


class TestMapPositionsBack:
    """Tests for map_positions_back function."""

    def test_basic_mapping(self):
        """Test basic position mapping."""
        text = "абв"
        _, position_map = transliterate_uzbek_cyrillic(text)
        # Map positions 0-3 (transliterated) back to original
        orig_start, orig_end = map_positions_back(0, 3, position_map, text)
        assert orig_start == 0
        assert orig_end == 3

    def test_multi_char_mapping(self):
        """Test mapping with multi-character transliterations."""
        text = "ёч"  # yo + ch
        _, position_map = transliterate_uzbek_cyrillic(text)
        # Transliterated: "yoch" (4 chars)
        # Map back from "yo" (positions 0-2)
        orig_start, orig_end = map_positions_back(0, 2, position_map, text)
        assert orig_start == 0
        assert orig_end == 1  # Just the ё character

    def test_out_of_bounds_handling(self):
        """Test handling of out-of-bounds positions."""
        text = "аб"
        _, position_map = transliterate_uzbek_cyrillic(text)
        # Test with positions beyond the map
        orig_start, orig_end = map_positions_back(100, 200, position_map, text)
        # Should clamp to valid positions
        assert orig_start >= 0
        assert orig_end <= len(text) + 1


class TestHasCyrillic:
    """Tests for has_cyrillic function."""

    def test_cyrillic_text(self):
        """Test text with Cyrillic characters."""
        assert has_cyrillic("Привет мир") is True
        assert has_cyrillic("абв") is True
        assert has_cyrillic("АБВГД") is True

    def test_latin_text(self):
        """Test text without Cyrillic characters."""
        assert has_cyrillic("Hello world") is False
        assert has_cyrillic("abc123") is False
        assert has_cyrillic("") is False

    def test_mixed_text(self):
        """Test mixed Cyrillic and Latin text."""
        assert has_cyrillic("Hello Мир") is True
        assert has_cyrillic("Test а test") is True

    def test_uzbek_cyrillic(self):
        """Test Uzbek-specific Cyrillic characters."""
        assert has_cyrillic("ўҳқғ") is True
        assert has_cyrillic("Ўзбекистон") is True


class TestDedupePresidioEntities:
    """Tests for dedupe_presidio_entities function."""

    def test_no_overlap(self):
        """Test entities without overlap."""
        entities = [
            {"start": 0, "end": 5, "type": "email", "text": "a@b.c"},
            {"start": 10, "end": 15, "type": "phone", "text": "12345"},
        ]
        result = dedupe_presidio_entities(entities)
        assert len(result) == 2

    def test_email_priority_over_url(self):
        """Test that email takes priority over URL for overlapping spans."""
        entities = [
            {"start": 0, "end": 15, "type": "url", "text": "test@example.com"},
            {"start": 0, "end": 15, "type": "email", "text": "test@example.com"},
        ]
        result = dedupe_presidio_entities(entities)
        assert len(result) == 1
        assert result[0]["type"] == "email"

    def test_partial_overlap(self):
        """Test partial overlap handling."""
        entities = [
            {"start": 0, "end": 10, "type": "email", "text": "test@ex.co"},
            {"start": 5, "end": 15, "type": "url", "text": "ex.com/path"},
        ]
        result = dedupe_presidio_entities(entities)
        # Email should be kept due to priority
        assert len(result) == 1
        assert result[0]["type"] == "email"

    def test_longer_span_preferred(self):
        """Test that longer spans are preferred for same type."""
        entities = [
            {"start": 0, "end": 5, "type": "phone", "text": "12345"},
            {"start": 0, "end": 10, "type": "phone", "text": "1234567890"},
        ]
        result = dedupe_presidio_entities(entities)
        assert len(result) == 1
        assert result[0]["end"] == 10


class TestMergeEntities:
    """Tests for merge_entities function."""

    def test_empty_inputs(self):
        """Test with empty inputs."""
        assert merge_entities([], []) == []

    def test_only_rubai_entities(self):
        """Test with only rubai entities."""
        rubai = [
            {
                "start": 0,
                "end": 5,
                "type": "person",
                "text": "Alisher",
                "source": "rubai",
            },
        ]
        result = merge_entities(rubai, [])
        assert len(result) == 1
        assert result[0]["source"] == "rubai"

    def test_only_presidio_entities(self):
        """Test with only presidio entities."""
        presidio = [
            {
                "start": 0,
                "end": 15,
                "type": "email",
                "text": "test@example.com",
                "source": "presidio",
            },
        ]
        result = merge_entities([], presidio)
        assert len(result) == 1
        assert result[0]["source"] == "presidio"

    def test_presidio_priority_types_win(self):
        """Test that presidio priority types take precedence."""
        rubai = [
            {
                "start": 0,
                "end": 15,
                "type": "person",
                "text": "test@example.com",
                "source": "rubai",
            },
        ]
        presidio = [
            {
                "start": 0,
                "end": 15,
                "type": "email",
                "text": "test@example.com",
                "source": "presidio",
            },
        ]
        result = merge_entities(rubai, presidio)
        # Email (presidio priority) should win
        assert len(result) == 1
        assert result[0]["type"] == "email"
        assert result[0]["source"] == "presidio"

    def test_rubai_fills_gaps(self):
        """Test that rubai entities fill gaps around presidio."""
        rubai = [
            {
                "start": 0,
                "end": 7,
                "type": "person",
                "text": "Alisher",
                "source": "rubai",
            },
            {
                "start": 30,
                "end": 40,
                "type": "location",
                "text": "Tashkent",
                "source": "rubai",
            },
        ]
        presidio = [
            {
                "start": 10,
                "end": 25,
                "type": "email",
                "text": "test@example.com",
                "source": "presidio",
            },
        ]
        result = merge_entities(rubai, presidio)
        assert len(result) == 3
        # Should be sorted by start position
        assert result[0]["start"] == 0
        assert result[1]["start"] == 10
        assert result[2]["start"] == 30

    def test_non_priority_presidio_added(self):
        """Test that non-priority presidio entities are added if no overlap."""
        rubai = [
            {
                "start": 0,
                "end": 7,
                "type": "person",
                "text": "Alisher",
                "source": "rubai",
            },
        ]
        presidio = [
            {
                "start": 20,
                "end": 30,
                "type": "ip_address",
                "text": "192.168.1.1",
                "source": "presidio",
            },
        ]
        result = merge_entities(rubai, presidio)
        assert len(result) == 2

    def test_result_sorted_by_start(self):
        """Test that results are sorted by start position."""
        rubai = [
            {
                "start": 50,
                "end": 60,
                "type": "person",
                "text": "Person",
                "source": "rubai",
            },
            {
                "start": 10,
                "end": 20,
                "type": "location",
                "text": "Place",
                "source": "rubai",
            },
        ]
        presidio = [
            {
                "start": 30,
                "end": 40,
                "type": "email",
                "text": "a@b.com",
                "source": "presidio",
            },
        ]
        result = merge_entities(rubai, presidio)
        # Should be sorted: 10, 30, 50
        positions = [e["start"] for e in result]
        assert positions == sorted(positions)


class TestConstants:
    """Tests for module constants."""

    def test_cyrillic_to_latin_completeness(self):
        """Test that CYRILLIC_TO_LATIN has both lower and upper case."""
        # Check some key mappings
        assert CYRILLIC_TO_LATIN["а"] == "a"
        assert CYRILLIC_TO_LATIN["А"] == "A"
        # Uzbek-specific
        assert CYRILLIC_TO_LATIN["ў"] == "o'"
        assert CYRILLIC_TO_LATIN["Ў"] == "O'"

    def test_presidio_entities_set(self):
        """Test PRESIDIO_ENTITIES contains expected types."""
        assert "EMAIL_ADDRESS" in PRESIDIO_ENTITIES
        assert "PHONE_NUMBER" in PRESIDIO_ENTITIES
        assert "CREDIT_CARD" in PRESIDIO_ENTITIES
        assert "IP_ADDRESS" in PRESIDIO_ENTITIES

    def test_presidio_type_map_consistency(self):
        """Test PRESIDIO_TYPE_MAP maps all PRESIDIO_ENTITIES."""
        for entity_type in PRESIDIO_ENTITIES:
            assert entity_type in PRESIDIO_TYPE_MAP

    def test_presidio_priority_types(self):
        """Test PRESIDIO_PRIORITY_TYPES contains expected types."""
        assert "url" in PRESIDIO_PRIORITY_TYPES
        assert "phone" in PRESIDIO_PRIORITY_TYPES
        assert "email" in PRESIDIO_PRIORITY_TYPES
