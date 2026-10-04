from fastapi.testclient import TestClient

from app.main import app


def test_tool_timeout_fault_retries_without_crash():
    with TestClient(app) as client:
        response = client.post(
            "/arena/run",
            json={
                "task": (
                    "I have PDC assignment due Tuesday needing 4 hours and "
                    "AI quiz Wednesday needing 3 hours. Classes Monday 8:30-14:30 "
                    "and Tuesday 10:00-13:00. Make a study plan."
                ),
                "arena_config": {"max_steps": 6, "fault": "tool_timeout"},
            },
        )
        assert response.status_code == 200
        body = response.json()
        assert body["status"] in {"completed", "tool_error"}
        outcomes = [t["outcome"] for t in body["tool_calls"]]
        assert "timeout" in outcomes
        # Server must remain healthy after the faulted run.
        assert client.get("/health").json()["status"] == "ok"
