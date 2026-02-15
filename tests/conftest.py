"""Pytest configuration and fixtures for mocking models."""

from unittest.mock import MagicMock, patch

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

    # Mock NLP engine
    mock_engine = MagicMock()
    mock_provider = MagicMock()
    mock_provider.return_value.create_engine.return_value = mock_engine

    # Patch at both the source module and where it's imported in src.main
    with (
        patch("transformers.pipeline", return_value=mock_ner),
        patch("src.main.pipeline", return_value=mock_ner),
        patch("presidio_analyzer.AnalyzerEngine", return_value=mock_presidio),
        patch("src.main.AnalyzerEngine", return_value=mock_presidio),
        patch("presidio_analyzer.nlp_engine.NlpEngineProvider", mock_provider),
        patch("src.main.NlpEngineProvider", mock_provider),
        patch("src.main.RecognizerRegistry", MagicMock()),
    ):
        yield {
            "ner_pipeline": mock_ner,
            "presidio_analyzer": mock_presidio,
        }
