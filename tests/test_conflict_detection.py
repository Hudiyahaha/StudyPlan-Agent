from app.tools import check_conflicts, generate_study_schedule, validate_study_input


def test_validate_flags_missing_deadline():
    result = validate_study_input(
        {
            "tasks": [{"name": "AI Quiz", "estimated_hours": 2, "priority": "high"}],
            "classes": [],
        }
    )
    assert result["complete"] is False
    assert any("deadline" in issue.lower() for issue in result["issues"])


def test_conflict_detection_finds_class_overlap():
    result = check_conflicts(
        {
            "schedule": [
                {
                    "task_name": "AI Quiz",
                    "day": "Monday",
                    "start": "09:00",
                    "end": "10:30",
                    "duration_hours": 1.5,
                }
            ],
            "classes": [{"day": "Monday", "start": "08:30", "end": "14:30", "title": "Class"}],
            "unavailable": [],
        }
    )
    assert result["ok"] is False
    assert result["conflict_count"] >= 1
    assert result["conflicts"][0]["type"] == "class_overlap"


def test_scheduler_avoids_class_hours():
    result = generate_study_schedule(
        {
            "tasks": [
                {
                    "name": "PDC Assignment",
                    "deadline": "2026-10-06T23:59:00",
                    "estimated_hours": 2,
                    "priority": "high",
                }
            ],
            "classes": [{"day": "Monday", "start": "08:30", "end": "14:30", "title": "Class"}],
            "unavailable": [],
            "preferred_hours": {"start": "16:00", "end": "22:00"},
        }
    )
    assert result["ok"] is True
    assert result["schedule"]
    for session in result["schedule"]:
        if session["day"] == "Monday":
            assert session["start"] >= "14:30" or session["end"] <= "08:30" or session["start"] >= "16:00"
