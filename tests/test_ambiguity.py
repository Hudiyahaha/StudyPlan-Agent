from fastapi.testclient import TestClient

from app.main import app


def test_ambiguous_exam_request_asks_clarification():
    with TestClient(app) as client:
        response = client.post(
            "/arena/run",
            json={
                "task": "I have an exam soon. Make me a study plan.",
                "arena_config": {"max_steps": 6, "fault": "none"},
            },
        )
        assert response.status_code == 200
        body = response.json()
        assert body["status"] == "needs_clarification"
        assert body["stop_reason"] == "needs_clarification"
        # Must not invent a concrete dated plan.
        assert "2026-" not in body["final_response"]
