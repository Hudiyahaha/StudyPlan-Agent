from fastapi.testclient import TestClient

from app.main import app


def test_clarification_then_continue():
    with TestClient(app) as client:
        session = "clarify-session-0001"
        first = client.post(
            "/chat",
            json={
                "session_id": session,
                "task": "Make me a plan for my AI assignment.",
                "model": "studyplan-mock-v1",
            },
        )
        assert first.status_code == 200
        assert first.json()["status"] == "needs_clarification"

        second = client.post(
            "/chat",
            json={
                "session_id": session,
                "task": "Tuesday, around 4 hours.",
                "model": "studyplan-mock-v1",
            },
        )
        assert second.status_code == 200
        body = second.json()
        assert body["status"] in {"completed", "needs_clarification"}
        # With deadline + hours, mock should complete a plan.
        assert body["status"] == "completed"
        assert body["stop_reason"] == "goal_completed"
