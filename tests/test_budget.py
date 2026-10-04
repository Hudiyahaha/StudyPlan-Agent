from fastapi.testclient import TestClient

from app.main import app


def test_budget_exceeded_with_low_max_steps():
    with TestClient(app) as client:
        response = client.post(
            "/arena/run",
            json={
                "task": (
                    "I have PDC assignment due Tuesday needing 4 hours and "
                    "AI quiz Wednesday needing 3 hours. Classes Monday 8:30-14:30 "
                    "and Tuesday 10:00-13:00. Make a study plan."
                ),
                "arena_config": {"max_steps": 1, "fault": "none"},
            },
        )
        assert response.status_code == 200
        body = response.json()
        assert body["status"] == "budget_exceeded"
        assert body["stop_reason"] == "max_steps_exceeded"
        assert body["steps"] <= 1
