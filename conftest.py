"""Pytest configuration and fixtures for mocking models."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest


@pytest.fixture(scope="session", autouse=True)
def mock_models():
    """Mock the heavy ML models to prevent loading during tests."""
    # Mock transformers pipeline
    mock_ner = MagicMock()
    mock_ner.return_value = []

    # Mock Presidio analyzer
    mock_presidio = MagicMock()
    mock_presidio.analyze = MagicMock(return_value=[])

    with (
        patch("transformers.pipeline", return_value=mock_ner),
        patch("presidio_analyzer.AnalyzerEngine", return_value=mock_presidio),
        patch("presidio_analyzer.nlp_engine.NlpEngineProvider") as mock_provider,
    ):
        # Mock the NLP engine provider
        mock_engine = MagicMock()
        mock_provider.return_value.create_engine.return_value = mock_engine

        yield {
            "ner_pipeline": mock_ner,
            "presidio_analyzer": mock_presidio,
        }
