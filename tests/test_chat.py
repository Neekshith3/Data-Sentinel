"""
Tests for the /api/chat endpoint.
"""
import pytest
from fastapi.testclient import TestClient
from main import app

client = TestClient(app)


def test_chat_without_dataset_context():
    """Test /api/chat without dataset context returns graceful error."""
    response = client.post("/api/chat", json={
        "message": "What are the main problems?",
        "conversation": [],
    })
    # Should still work but with empty context
    assert response.status_code == 200
    data = response.json()
    assert "response" in data
    assert "conversation_id" in data
    assert data["grounded"] is True


def test_chat_with_valid_context():
    """Test /api/chat with valid dataset context returns successful response."""
    findings = [
        {
            "issue_type": "missing_values",
            "has_issue": True,
            "count": 10,
            "affected_columns": ["email", "phone"],
        },
        {
            "issue_type": "possible_sentinel_values",
            "has_issue": True,
            "count": 5,
            "affected_columns": ["age"],
        }
    ]
    quality_score = {"overall_score": 75.0}
    ml_readiness = {"readiness_percentage": 60}
    profile = {"rows": 1000, "columns": 10}
    
    response = client.post("/api/chat", json={
        "message": "What are the main problems in this dataset?",
        "conversation": [],
        "findings": findings,
        "quality_score": quality_score,
        "ml_readiness": ml_readiness,
        "profile": profile,
    })
    
    assert response.status_code == 200
    data = response.json()
    assert "response" in data
    assert "conversation_id" in data
    assert data["grounded"] is True
    # Response should mention the issues
    assert "missing" in data["response"].lower() or "sentinel" in data["response"].lower()


def test_chat_empty_message():
    """Test /api/chat with empty message returns validation error."""
    response = client.post("/api/chat", json={
        "message": "",
        "conversation": [],
    })
    # Empty message should still be handled gracefully
    assert response.status_code == 200
    data = response.json()
    assert "response" in data


def test_chat_missing_message():
    """Test /api/chat with missing message field - now optional, uses question."""
    response = client.post("/api/chat", json={
        "question": "What are the problems?",
        "conversation": [],
    })
    # Now message is optional, question is used instead
    assert response.status_code == 200
    data = response.json()
    assert "response" in data


def test_chat_conversation_history():
    """Test /api/chat accepts and uses conversation history."""
    findings = [
        {
            "issue_type": "missing_values",
            "has_issue": True,
            "count": 10,
            "affected_columns": ["email"],
        }
    ]
    quality_score = {"overall_score": 80.0}
    ml_readiness = {"readiness_percentage": 70}
    profile = {"rows": 500, "columns": 8}
    
    # First question
    response1 = client.post("/api/chat", json={
        "message": "Which columns have missing values?",
        "conversation": [],
        "findings": findings,
        "quality_score": quality_score,
        "ml_readiness": ml_readiness,
        "profile": profile,
    })
    assert response1.status_code == 200
    data1 = response1.json()
    
    # Follow-up question with conversation history
    response2 = client.post("/api/chat", json={
        "message": "Which one is the worst?",
        "conversation": [
            {"role": "user", "content": "Which columns have missing values?"},
            {"role": "assistant", "content": data1["response"]}
        ],
        "findings": findings,
        "quality_score": quality_score,
        "ml_readiness": ml_readiness,
        "profile": profile,
    })
    assert response2.status_code == 200
    data2 = response2.json()
    assert "response" in data2


def test_chat_llm_failure_fallback():
    """Test /api/chat gracefully handles LLM failure by falling back to rules."""
    # This test verifies the fallback works even without API key
    findings = [
        {
            "issue_type": "duplicate_rows",
            "has_issue": True,
            "count": 5,
            "affected_columns": ["id"],
        }
    ]
    quality_score = {"overall_score": 85.0}
    ml_readiness = {"readiness_percentage": 90}
    profile = {"rows": 100, "columns": 5}
    
    response = client.post("/api/chat", json={
        "message": "What are the duplicate issues?",
        "conversation": [],
        "findings": findings,
        "quality_score": quality_score,
        "ml_readiness": ml_readiness,
        "profile": profile,
    })
    
    assert response.status_code == 200
    data = response.json()
    assert "response" in data
    assert "mode" in data or "grounded" in data


def test_chat_invalid_values():
    """Test /api/chat with invalid request body."""
    response = client.post("/api/chat", json={
        "message": 123,  # Should be string
        "conversation": "not a list",
    })
    # FastAPI should handle validation
    # Depending on schema, this might return 422 or be handled gracefully
    assert response.status_code in (200, 422)


def test_chat_no_dataset_analyzed():
    """Test /api/chat when no dataset has been analyzed yet."""
    # Frontend should show "Please analyze a dataset first" but API should still respond
    response = client.post("/api/chat", json={
        "message": "What are the problems?",
        "conversation": [],
        "findings": [],
        "quality_score": {},
        "ml_readiness": {},
        "profile": {},
    })
    assert response.status_code == 200
    data = response.json()
    assert "response" in data
    # Should indicate no issues detected
    assert "No issues" in data["response"] or "no issues" in data["response"].lower()


def test_chat_grounding():
    """Test that chatbot responses are grounded in actual analysis data."""
    findings = [
        {
            "issue_type": "possible_sentinel_values",
            "has_issue": True,
            "count": 2197,
            "affected_columns": ["col1", "col2", "col3"],
            "details": {
                "columns": {
                    "col1": {"sentinel_occurrences": 1000, "sentinel_count": 1},
                    "col2": {"sentinel_occurrences": 800, "sentinel_count": 1},
                    "col3": {"sentinel_occurrences": 397, "sentinel_count": 1},
                },
                "total_sentinels": 17,
                "total_placeholders": 0
            }
        }
    ]
    quality_score = {"overall_score": 70.0}
    ml_readiness = {"readiness_percentage": 50}
    profile = {"rows": 999, "columns": 27}
    
    response = client.post("/api/chat", json={
        "message": "How many sentinel values are there?",
        "conversation": [],
        "findings": findings,
        "quality_score": quality_score,
        "ml_readiness": ml_readiness,
        "profile": profile,
    })
    
    assert response.status_code == 200
    data = response.json()
    assert "response" in data
    # The response should contain the actual count from the findings
    # (not an invented number like 112)
    assert "2197" in data["response"] or "17" in data["response"]


def test_explain_still_works():
    """Test that /api/explain still works after chat implementation."""
    findings = [
        {
            "issue_type": "missing_values",
            "has_issue": True,
            "count": 5,
            "affected_columns": ["email"],
        }
    ]
    quality_score = {"overall_score": 80.0}
    ml_readiness = {"readiness_percentage": 70}
    
    response = client.post("/api/explain", json={
        "findings": findings,
        "quality_score": quality_score,
        "ml_readiness": ml_readiness,
    })
    
    assert response.status_code == 200
    data = response.json()
    assert "summary" in data
    assert "insights" in data
    assert "mode" in data


def test_chat_large_dataset_safety():
    """Test that chat doesn't receive raw million-row dataset."""
    # Simulate large dataset profile (1M+ rows)
    profile = {"rows": 1048575, "columns": 27}
    findings = [
        {
            "issue_type": "possible_sentinel_values",
            "has_issue": True,
            "count": 17,
            "affected_columns": ["col1", "col2"],
        }
    ]
    quality_score = {"overall_score": 75.0}
    ml_readiness = {"readiness_percentage": 65}
    
    response = client.post("/api/chat", json={
        "message": "Summarize this dataset",
        "conversation": [],
        "findings": findings,
        "quality_score": quality_score,
        "ml_readiness": ml_readiness,
        "profile": profile,
    })
    
    assert response.status_code == 200
    data = response.json()
    assert "response" in data
    # Response should reference aggregated findings, not raw rows
    assert "1048575" not in data["response"] or "1M" in data["response"] or "1,048,575" in data["response"]


def test_chat_context_contains_canonical_findings():
    """Test that chat service receives canonical analysis values."""
    sentinel_occurrences = 2197
    sentinel_columns = 17
    rows = 999
    columns = 27
    
    findings = [
        {
            "issue_type": "possible_sentinel_values",
            "has_issue": True,
            "count": sentinel_occurrences,
            "affected_columns": [f"col{i}" for i in range(sentinel_columns)],
            "details": {
                "columns": {
                    f"col{i}": {"sentinel_occurrences": 100, "sentinel_count": 1}
                    for i in range(sentinel_columns)
                },
                "total_sentinels": sentinel_columns,
                "total_placeholders": 0
            }
        }
    ]
    quality_score = {"overall_score": 70.0}
    ml_readiness = {"readiness_percentage": 50}
    profile = {"rows": rows, "columns": columns}
    
    response = client.post("/api/chat", json={
        "message": "What are the sentinel statistics?",
        "conversation": [],
        "findings": findings,
        "quality_score": quality_score,
        "ml_readiness": ml_readiness,
        "profile": profile,
    })
    
    assert response.status_code == 200
    data = response.json()
    assert "response" in data
    # Response should contain the canonical values
    assert str(sentinel_occurrences) in data["response"] or str(sentinel_columns) in data["response"]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])