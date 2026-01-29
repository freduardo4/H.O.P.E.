import pytest
import os
import sys
from unittest.mock import MagicMock, patch

# Add src/ai-training to path to import hope_ai
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from hope_ai.rag_server import RAGService

@pytest.fixture
def temp_docs(tmp_path):
    """Creates temporary documentation files for testing."""
    d = tmp_path / "docs"
    d.mkdir()
    
    (d / "doc1.md").write_text("KWP2000 is a communications protocol used for on-board vehicle diagnostics systems (OBD).", encoding="utf-8")
    (d / "doc2.md").write_text("UDS (Unified Diagnostic Services) is an ISO standard specified in ISO 14229.", encoding="utf-8")
    (d / "doc3.txt").write_text("ECU tuning involves adjusting the mapping of the engine control unit.", encoding="utf-8")
    
    return str(d)

def test_rag_loading(temp_docs):
    rag = RAGService([temp_docs])
    rag.load_documents()
    
    assert len(rag.documents) == 3
    assert rag.vectorizer is not None

def test_rag_retrieval(temp_docs):
    rag = RAGService([temp_docs])
    rag.load_documents()
    
    # Text exact match
    result = rag.query("What is KWP2000?")
    assert "results" in result
    assert len(result["results"]) > 0
    assert "KWP2000" in result["results"][0]["content"]
    
    # Test concept matching
    result = rag.query("ISO standard")
    assert "UDS" in result["results"][0]["content"]
    
    # Test irrelevant
    result = rag.query("Banana")
    # Should get nothing or low score, but our threshold is 0.05 so might just trigger 0 results
    # or low results.
    # Actually "Banana" won't match any tokens if not in vocab.
    if "results" in result:
       assert len(result["results"]) == 0 or result["results"][0]["score"] < 0.1

def test_rag_empty_query(temp_docs):
    rag = RAGService([temp_docs])
    rag.load_documents()
    
    # Just ensure it doesn't crash
    rag.query("")
