from fastapi.testclient import TestClient

from app.main import app


def test_invalid_contract_fault_recovers_or_errors():
    with TestClient(app) as client:
        response = client.post(
            "/arena/run",
            json={
                "task": (
                    "I have PDC assignment due Tuesday needing 4 hours and "
                    "AI quiz Wednesday needing 3 hours. Classes Monday 8:30-14:30 "
                    "and Tuesday 10:00-13:00. Make a study plan."
                ),
                "arena_config": {"max_steps": 6, "fault": "invalid_agent_decision"},
            },
        )
        assert response.status_code == 200
        body = response.json()
        # First decision is injected as invalid; bounded repair should still finish.
        assert body["status"] in {"completed", "contract_error"}
        if body["status"] == "contract_error":
            assert body["stop_reason"] == "contract_validation_failed"
        else:
            assert body["stop_reason"] == "goal_completed"
            assert any(e.get("event") == "decision_rejected" for e in body["events"])
