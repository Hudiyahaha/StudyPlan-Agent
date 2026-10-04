from fastapi.testclient import TestClient

from app.main import app

COMPLETE_TASK = (
    "I have PDC assignment due Tuesday needing 4 hours and "
    "AI quiz Wednesday needing 3 hours. Classes Monday 8:30-14:30 "
    "and Tuesday 10:00-13:00. Make a study plan."
)


def test_arena_run_completed():
    with TestClient(app) as client:
        response = client.post(
            "/arena/run",
            json={"task": COMPLETE_TASK, "external_context": [], "arena_config": {"max_steps": 6, "fault": "none"}},
        )
        assert response.status_code == 200
        body = response.json()
        assert body["status"] == "completed"
        assert body["stop_reason"] == "goal_completed"
        assert body["steps"] >= 3
        tools = [t["tool"] for t in body["tool_calls"]]
        assert "validate_study_input" in tools
        assert "generate_study_schedule" in tools
        assert "summarize_plan" in tools
        assert "study plan" in body["final_response"].lower() or "Monday" in body["final_response"] or "Tuesday" in body["final_response"]
