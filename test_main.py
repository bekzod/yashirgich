"""Tests for main.py - FastAPI endpoints and PII masking."""

import time
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from main import (
    InMemoryReplacementStore,
    MaskedEntity,
    PIIResponse,
    process_messages_for_pii,
    restore_pii,
)


class TestInMemoryReplacementStore:
    """Tests for InMemoryReplacementStore."""

    @pytest.fixture
    def store(self):
        return InMemoryReplacementStore()

    @pytest.mark.asyncio
    async def test_save_and_get(self, store):
        """Test basic save and get operations."""
        replacement_map = {"<<EMAIL_1>>": "test@example.com"}
        await store.save("request-1", replacement_map)

        result = await store.get("request-1")
        assert result == replacement_map

    @pytest.mark.asyncio
    async def test_get_nonexistent(self, store):
        """Test getting a nonexistent key."""
        result = await store.get("nonexistent")
        assert result is None

    @pytest.mark.asyncio
    async def test_delete(self, store):
        """Test delete operation."""
        await store.save("request-1", {"key": "value"})
        await store.delete("request-1")

        result = await store.get("request-1")
        assert result is None

    @pytest.mark.asyncio
    async def test_delete_nonexistent(self, store):
        """Test deleting a nonexistent key (should not raise)."""
        await store.delete("nonexistent")  # Should not raise

    @pytest.mark.asyncio
    async def test_ttl_expiry(self, store):
        """Test that entries expire after TTL."""
        await store.save("request-1", {"key": "value"}, ttl=1)

        # Should exist immediately
        result = await store.get("request-1")
        assert result is not None

        # Wait for expiry
        time.sleep(1.1)

        result = await store.get("request-1")
        assert result is None


class TestRestorePii:
    """Tests for restore_pii function."""

    def test_basic_restore(self):
        """Test basic PII restoration."""
        text = "Hello <<EMAIL_1>>, your phone is <<PHONE_1>>"
        replacement_map = {
            "<<EMAIL_1>>": "test@example.com",
            "<<PHONE_1>>": "+1234567890",
        }

        result = restore_pii(text, replacement_map)
        assert result == "Hello test@example.com, your phone is +1234567890"

    def test_empty_replacement_map(self):
        """Test with empty replacement map."""
        text = "Hello world"
        result = restore_pii(text, {})
        assert result == "Hello world"

    def test_no_matches(self):
        """Test when replacement map keys don't match text."""
        text = "Hello world"
        replacement_map = {"<<EMAIL_1>>": "test@example.com"}
        result = restore_pii(text, replacement_map)
        assert result == "Hello world"

    def test_multiple_occurrences(self):
        """Test that all occurrences are replaced."""
        text = "Email: <<EMAIL_1>>, confirm: <<EMAIL_1>>"
        replacement_map = {"<<EMAIL_1>>": "test@example.com"}

        result = restore_pii(text, replacement_map)
        assert result == "Email: test@example.com, confirm: test@example.com"


class TestProcessMessagesForPii:
    """Tests for process_messages_for_pii function."""

    @patch("main.mask_text_simple")
    def test_string_content(self, mock_mask):
        """Test processing messages with string content."""
        mock_mask.return_value = ("masked text", {"<<EMAIL_1>>": "test@example.com"})

        messages = [{"role": "user", "content": "test@example.com"}]
        processed, combined_map = process_messages_for_pii(messages)

        assert processed[0]["content"] == "masked text"
        assert "<<EMAIL_1>>" in combined_map

    @patch("main.mask_text_simple")
    def test_multimodal_content(self, mock_mask):
        """Test processing messages with multimodal content."""
        mock_mask.return_value = ("masked text", {"<<EMAIL_1>>": "test@example.com"})

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "test@example.com"},
                    {
                        "type": "image_url",
                        "image_url": {"url": "http://example.com/img.png"},
                    },
                ],
            }
        ]

        processed, combined_map = process_messages_for_pii(messages)

        assert processed[0]["content"][0]["text"] == "masked text"
        assert processed[0]["content"][1]["type"] == "image_url"

    @patch("main.mask_text_simple")
    def test_multiple_messages(self, mock_mask):
        """Test processing multiple messages."""
        call_count = [0]

        def mock_impl(text):
            call_count[0] += 1
            return (f"masked_{call_count[0]}", {f"<<ENTITY_{call_count[0]}>>": text})

        mock_mask.side_effect = mock_impl

        messages = [
            {"role": "user", "content": "message1"},
            {"role": "assistant", "content": "message2"},
        ]

        processed, combined_map = process_messages_for_pii(messages)

        assert len(processed) == 2
        assert len(combined_map) == 2

    def test_empty_messages(self):
        """Test with empty messages list."""
        processed, combined_map = process_messages_for_pii([])
        assert processed == []
        assert combined_map == {}

    @patch("main.mask_text_simple")
    def test_non_string_non_list_content(self, mock_mask):
        """Test that non-string, non-list content is passed through."""
        messages = [{"role": "user", "content": None}]
        processed, _ = process_messages_for_pii(messages)

        assert processed[0]["content"] is None
        mock_mask.assert_not_called()


class TestMaskedEntity:
    """Tests for MaskedEntity Pydantic model."""

    def test_create_masked_entity(self):
        """Test creating a MaskedEntity."""
        entity = MaskedEntity(
            original="test@example.com",
            masked_as="<<EMAIL_1>>",
            entity_type="EMAIL",
            start=0,
            end=16,
            source="presidio",
        )

        assert entity.original == "test@example.com"
        assert entity.masked_as == "<<EMAIL_1>>"
        assert entity.entity_type == "EMAIL"
        assert entity.start == 0
        assert entity.end == 16
        assert entity.source == "presidio"


class TestPIIResponse:
    """Tests for PIIResponse Pydantic model."""

    def test_create_pii_response(self):
        """Test creating a PIIResponse."""
        entity = MaskedEntity(
            original="test@example.com",
            masked_as="<<EMAIL_1>>",
            entity_type="EMAIL",
            start=0,
            end=16,
            source="presidio",
        )

        response = PIIResponse(
            masked_text="Hello <<EMAIL_1>>",
            entities=[entity],
        )

        assert response.masked_text == "Hello <<EMAIL_1>>"
        assert len(response.entities) == 1

    def test_empty_entities(self):
        """Test PIIResponse with no entities."""
        response = PIIResponse(
            masked_text="Hello world",
            entities=[],
        )

        assert response.masked_text == "Hello world"
        assert len(response.entities) == 0


# Integration tests that require the app to be running with models loaded
# These are marked to be skipped by default since they require heavy dependencies


@pytest.fixture
def client():
    """Create a test client with mocked models."""
    from main import app

    # We need to mock the global models since they're loaded in lifespan
    with (
        patch("main.ner_pipeline") as mock_ner,
        patch("main.presidio_analyzer") as mock_presidio,
        patch("main.replacement_store") as mock_store,
    ):
        mock_ner.return_value = []
        mock_presidio.analyze.return_value = []
        mock_store.save = AsyncMock()
        mock_store.get = AsyncMock(return_value={})
        mock_store.delete = AsyncMock()

        with TestClient(app, raise_server_exceptions=False) as client:
            yield client


class TestHealthEndpoint:
    """Tests for /health endpoint."""

    def test_health_models_not_loaded(self):
        """Test health check when models are not loaded."""
        import main

        # Save original values
        original_ner = main.ner_pipeline
        original_presidio = main.presidio_analyzer
        original_store = main.replacement_store

        try:
            # Set to None to simulate models not loaded
            main.ner_pipeline = None
            main.presidio_analyzer = None
            main.replacement_store = MagicMock()

            # Create a fresh app without lifespan to test the health endpoint directly
            from fastapi import FastAPI

            test_app = FastAPI()

            @test_app.get("/health")
            async def health():
                return await main.health_check()

            with TestClient(test_app, raise_server_exceptions=False) as client:
                response = client.get("/health")
                assert response.status_code == 503
        finally:
            # Restore original values
            main.ner_pipeline = original_ner
            main.presidio_analyzer = original_presidio
            main.replacement_store = original_store

    def test_health_models_loaded(self):
        """Test health check when models are loaded."""
        import main

        # Save original values
        original_ner = main.ner_pipeline
        original_presidio = main.presidio_analyzer
        original_store = main.replacement_store

        try:
            # Set to mock objects to simulate models loaded
            main.ner_pipeline = MagicMock()
            main.presidio_analyzer = MagicMock()
            main.replacement_store = MagicMock()

            # Create a fresh app without lifespan to test the health endpoint directly
            from fastapi import FastAPI

            test_app = FastAPI()

            @test_app.get("/health")
            async def health():
                return await main.health_check()

            with TestClient(test_app, raise_server_exceptions=False) as client:
                response = client.get("/health")
                assert response.status_code == 200
                data = response.json()
                assert data["status"] == "healthy"
        finally:
            # Restore original values
            main.ner_pipeline = original_ner
            main.presidio_analyzer = original_presidio
            main.replacement_store = original_store


class TestDetectEndpoint:
    """Tests for /detect endpoint."""

    def test_detect_empty_text(self):
        """Test detect endpoint with empty text."""
        from main import app

        with (
            patch("main.ner_pipeline", MagicMock()),
            patch("main.presidio_analyzer", MagicMock()),
        ):
            with TestClient(app, raise_server_exceptions=False) as client:
                response = client.post("/detect", json={"text": ""})
                assert response.status_code == 400

    def test_detect_whitespace_only(self):
        """Test detect endpoint with whitespace only."""
        from main import app

        with (
            patch("main.ner_pipeline", MagicMock()),
            patch("main.presidio_analyzer", MagicMock()),
        ):
            with TestClient(app, raise_server_exceptions=False) as client:
                response = client.post("/detect", json={"text": "   "})
                assert response.status_code == 400


class TestUploadEndpoint:
    """Tests for /upload endpoint."""

    def test_upload_empty_file(self):
        """Test upload endpoint with empty file."""
        from main import app

        with (
            patch("main.ner_pipeline", MagicMock(return_value=[])),
            patch("main.presidio_analyzer") as mock_presidio,
        ):
            mock_presidio.analyze.return_value = []

            with TestClient(app, raise_server_exceptions=False) as client:
                response = client.post(
                    "/upload",
                    files={"file": ("test.txt", b"", "text/plain")},
                )
                assert response.status_code == 400


class TestProxyEndpoint:
    """Tests for /proxy/v1/chat/completions endpoint."""

    def test_proxy_no_api_key(self):
        """Test proxy endpoint without API key."""
        from main import app

        with (
            patch("main.ner_pipeline", MagicMock()),
            patch("main.presidio_analyzer", MagicMock()),
            patch("main.replacement_store", MagicMock()),
            patch.dict("os.environ", {}, clear=True),
        ):
            # Remove OPENAI_API_KEY if it exists
            import os

            env_backup = os.environ.get("OPENAI_API_KEY")
            if "OPENAI_API_KEY" in os.environ:
                del os.environ["OPENAI_API_KEY"]

            try:
                with TestClient(app, raise_server_exceptions=False) as client:
                    response = client.post(
                        "/proxy/v1/chat/completions",
                        json={"messages": [{"role": "user", "content": "Hello"}]},
                    )
                    assert response.status_code == 401
            finally:
                if env_backup:
                    os.environ["OPENAI_API_KEY"] = env_backup

    def test_proxy_no_messages(self):
        """Test proxy endpoint without messages."""
        from main import app

        with (
            patch("main.ner_pipeline", MagicMock()),
            patch("main.presidio_analyzer", MagicMock()),
            patch("main.replacement_store", MagicMock()),
        ):
            with TestClient(app, raise_server_exceptions=False) as client:
                response = client.post(
                    "/proxy/v1/chat/completions",
                    json={},
                    headers={"Authorization": "Bearer test-key"},
                )
                assert response.status_code == 400

    def test_proxy_invalid_json(self):
        """Test proxy endpoint with invalid JSON."""
        from main import app

        with (
            patch("main.ner_pipeline", MagicMock()),
            patch("main.presidio_analyzer", MagicMock()),
            patch("main.replacement_store", MagicMock()),
        ):
            with TestClient(app, raise_server_exceptions=False) as client:
                response = client.post(
                    "/proxy/v1/chat/completions",
                    content="invalid json",
                    headers={
                        "Authorization": "Bearer test-key",
                        "Content-Type": "application/json",
                    },
                )
                assert response.status_code == 400
