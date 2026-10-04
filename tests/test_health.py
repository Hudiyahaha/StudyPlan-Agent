from fastapi.testclient import TestClient

from app.main import app


def test_health_ok():
    with TestClient(app) as client:
        response = client.get("/health")
        assert response.status_code == 200
        body = response.json()
        assert body["status"] == "ok"
        assert body["implementation"] == "studyplan"
        assert body["agent_name"] == "StudyPlan Agent"


def test_manifest():
    with TestClient(app) as client:
        response = client.get("/arena/manifest")
        assert response.status_code == 200
        body = response.json()
        assert body["agent_name"] == "StudyPlan Agent"
        assert body["domain"] == "Study-Plan Builder"
