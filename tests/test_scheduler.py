"""Scheduler and input-normalization rules. The clock is pinned to Monday 2026-10-05 08:00."""
import json

import pytest

from app.config import ROOT
from app.models import ClassSlot, StudySession
from app.tools import check_conflicts, generate_study_schedule, summarize_plan, validate_study_input

CLASSES = [
    {"day": "Monday", "start": "08:30", "end": "14:30"},
    {"day": "Tuesday", "start": "10:00", "end": "13:00"},
]


def task(name="PDC", deadline="2026-10-06", hours=4):
    return {"name": name, "deadline": deadline, "estimated_hours": hours}


def test_all_requested_hours_are_placed_when_time_exists():
    result = generate_study_schedule(
        {"tasks": [task(), task("AI Quiz", "2026-10-07", 3)], "classes": CLASSES}
    )
    assert result["warnings"] == [] and result["unscheduled_hours"] == {}
    placed = {}
    for session in result["schedule"]:
        placed[session["task_name"]] = placed.get(session["task_name"], 0) + session["duration_hours"]
    assert placed == {"PDC": 4, "AI Quiz": 3}
    assert check_conflicts({"schedule": result["schedule"], "classes": CLASSES})["ok"] is True


def test_blocks_are_bounded_and_separated_by_breaks():
    sessions = generate_study_schedule({"tasks": [task(hours=4)]})["schedule"]
    assert all(s["duration_hours"] <= 1.5 for s in sessions)
    same_day = [s for s in sessions if s["date"] == "2026-10-05"]
    assert all(left["end"] < right["start"] for left, right in zip(same_day, same_day[1:]))


def test_sessions_respect_the_deadline_time_and_day():
    result = generate_study_schedule({"tasks": [task(deadline="2026-10-06T17:00", hours=20)]})
    assert all(s["date"] <= "2026-10-06" for s in result["schedule"])
    assert all(s["end"] <= "17:00" for s in result["schedule"] if s["date"] == "2026-10-06")
    assert result["unscheduled_hours"]["PDC"] > 0
    assert "fit before the deadline" in result["warnings"][0]


def test_earlier_deadline_is_placed_first():
    result = generate_study_schedule(
        {"tasks": [task("Later", "2026-10-09", 1), task("Sooner", "2026-10-06", 1)]}
    )
    assert result["schedule"][0]["task_name"] == "Sooner"


def test_weekly_classes_recur_across_weeks():
    result = generate_study_schedule(
        {
            "tasks": [task(deadline="2026-10-13", hours=30)],
            "classes": [{"day": "Monday", "start": "16:00", "end": "22:00"}],
        }
    )
    dates = {s["date"] for s in result["schedule"]}
    assert "2026-10-06" in dates and "2026-10-05" not in dates and "2026-10-12" not in dates


def test_same_weekday_in_different_weeks_is_not_a_session_conflict():
    base = {"task_name": "A", "day": "Monday", "start": "16:00", "end": "17:00", "duration_hours": 1}
    schedule = [{**base, "date": "2026-10-05"}, {**base, "date": "2026-10-12"}]
    assert check_conflicts({"schedule": schedule})["ok"] is True
    assert check_conflicts({"schedule": [schedule[0], schedule[0]]})["ok"] is False


def test_times_and_days_are_normalized_before_comparison():
    slot = ClassSlot(day="mon", start="8:30", end="2:30 pm")
    assert (slot.day, slot.start, slot.end) == ("Monday", "08:30", "14:30")


@pytest.mark.parametrize("start,end", [("14:30", "8:30"), ("25:00", "26:00"), ("soon", "later")])
def test_invalid_class_times_are_clarification_issues(start, end):
    result = validate_study_input(
        {"tasks": [task()], "classes": [{"day": "Monday", "start": start, "end": end}]}
    )
    assert result["complete"] is False and "Class slot 1 is invalid" in result["issues"][0]


def test_past_deadline_and_bad_hours_are_flagged():
    past = validate_study_input({"tasks": [task(deadline="2026-09-01")]})
    assert "past" in past["issues"][0]
    assert validate_study_input({"tasks": [task(hours=0)]})["complete"] is False
    assert validate_study_input({"tasks": [task(hours=500)]})["complete"] is False
    assert validate_study_input({"tasks": [task(deadline="next week")]})["complete"] is False


def test_inverted_study_window_is_flagged():
    result = validate_study_input({"tasks": [task()], "preferred_hours": {"start": "22:00", "end": "16:00"}})
    assert result["complete"] is False


def test_session_contract_rejects_inverted_range():
    with pytest.raises(ValueError):
        StudySession(task_name="A", day="Monday", start="18:00", end="17:00", duration_hours=1)


def test_long_plans_are_shortened_explicitly():
    schedule = [
        {
            "task_name": "A very long task name for this test " * 2,
            "day": "Monday",
            "date": f"2026-10-{day:02d}",
            "start": "16:00",
            "end": "17:00",
            "duration_hours": 1,
        }
        for day in range(5, 30)
    ] * 2
    summary = summarize_plan({"schedule": schedule, "warnings": ["Keep me visible"]})["summary"]
    assert len(summary) <= 1900 and "omitted" in summary and "Keep me visible" in summary


def test_sample_data_is_schedulable():
    sample = json.loads((ROOT / "data" / "sample_data.json").read_text(encoding="utf-8"))
    assert validate_study_input(sample)["complete"] is True
    assert generate_study_schedule(sample)["warnings"] == []
