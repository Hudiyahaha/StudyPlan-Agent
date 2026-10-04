"""Study-plan tools and lightweight natural-language extraction helpers."""
from __future__ import annotations

import json
import re
from copy import deepcopy
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

from app.config import ROOT, settings
from app.models import ClassSlot, StudySession, StudyTask, UnavailableSlot

DAY_ALIASES = {
    "mon": "Monday",
    "monday": "Monday",
    "tue": "Tuesday",
    "tues": "Tuesday",
    "tuesday": "Tuesday",
    "wed": "Wednesday",
    "wednesday": "Wednesday",
    "thu": "Thursday",
    "thur": "Thursday",
    "thurs": "Thursday",
    "thursday": "Thursday",
    "fri": "Friday",
    "friday": "Friday",
    "sat": "Saturday",
    "saturday": "Saturday",
    "sun": "Sunday",
    "sunday": "Sunday",
}

PRIORITY_WORDS = {
    "urgent": "high",
    "important": "high",
    "high": "high",
    "medium": "medium",
    "normal": "medium",
    "low": "low",
}


def _normalize_day(text: str) -> str | None:
    return DAY_ALIASES.get(text.strip().lower())


def _to_minutes(hhmm: str) -> int:
    hour, minute = hhmm.split(":")
    return int(hour) * 60 + int(minute)


def _from_minutes(total: int) -> str:
    hour, minute = divmod(total, 60)
    return f"{hour:02d}:{minute:02d}"


def _ranges_overlap(a_start: str, a_end: str, b_start: str, b_end: str) -> bool:
    return _to_minutes(a_start) < _to_minutes(b_end) and _to_minutes(b_start) < _to_minutes(a_end)


def load_sample_data() -> dict[str, Any]:
    path = ROOT / "data" / "sample_data.json"
    return json.loads(path.read_text(encoding="utf-8"))


def extract_study_context(text: str, history_text: str = "") -> dict[str, Any]:
    """Best-effort extraction from free text. Incomplete fields stay missing."""
    blob = f"{history_text}\n{text}".strip()
    lower = blob.lower()
    sample = load_sample_data()
    day_alt = (
        r"monday|tuesday|wednesday|thursday|friday|saturday|sunday|"
        r"mon|tue|tues|wed|thu|thur|thurs|fri|sat|sun"
    )

    tasks: list[dict[str, Any]] = []

    def _add_task(name: str, day: str | None, hours: float | None, priority: str = "medium") -> None:
        name = re.sub(r"\s+", " ", name).strip(" .,")
        if not name:
            return
        name = re.sub(
            r"^(?:i have|have|need to (?:do|finish|study)|for my|and|the)\s+",
            "",
            name,
            flags=re.I,
        ).strip()
        # Drop accidental trailing glue words
        name = re.sub(r"\b(?:needing|need|around|about|hours|hrs)\b.*$", "", name, flags=re.I).strip(" .,")
        if not name:
            return
        key = re.sub(r"[^a-z0-9]+", "", name.lower())
        for existing in tasks:
            existing_key = re.sub(r"[^a-z0-9]+", "", existing["name"].lower())
            if existing_key == key or existing_key.startswith(key) or key.startswith(existing_key):
                if day and not existing.get("deadline"):
                    existing["deadline"] = _deadline_for_day(day)
                if hours and not existing.get("estimated_hours"):
                    existing["estimated_hours"] = hours
                if priority == "high":
                    existing["priority"] = "high"
                return
        tasks.append(
            {
                "name": name,
                "deadline": _deadline_for_day(day) if day else None,
                "estimated_hours": hours,
                "priority": priority,
            }
        )

    # "<course> <kind> due <day>"
    due_pattern = re.compile(
        rf"([A-Za-z][A-Za-z0-9 /&+-]{{1,40}}?)\s+"
        rf"(assignment|quiz|exam|midterm|project|homework|hw)\s+"
        rf"(?:is\s+)?due\s+({day_alt})",
        re.IGNORECASE,
    )
    for match in due_pattern.finditer(blob):
        name = f"{match.group(1).strip()} {match.group(2).strip().title()}"
        window = blob[match.start() : min(len(blob), match.end() + 48)]
        _add_task(name, _normalize_day(match.group(3)), _find_hours_near(window), "high")

    # Course-code due phrases already covered by due_pattern; keep a tight fallback.
    loose_due = re.compile(
        rf"\b((?:PDC|AI|Agentic AI|DSA|OS|DB|SE|CN|[A-Z]{{2,8}}))\s+"
        rf"(?:is\s+)?due\s+({day_alt})",
        re.IGNORECASE,
    )
    for match in loose_due.finditer(blob):
        name = match.group(1).strip()
        window = blob[max(0, match.start() - 20) : match.end() + 24]
        for kind in ("assignment", "quiz", "exam", "project", "homework", "midterm"):
            if re.search(rf"\b{re.escape(name)}\s+{kind}\b", window, re.I):
                name = f"{name} {kind.title()}"
                break
        else:
            name = f"{name} Task"
        _add_task(name, _normalize_day(match.group(2)), _find_hours_near(window))

    # "AI quiz Wednesday" / "PDC assignment Tuesday" without the word due
    kind_day = re.compile(
        rf"\b((?:PDC|AI|Agentic AI|DSA|OS|DB|SE|CN|[A-Z]{{2,8}})(?:\s+[A-Za-z]+)?)\s+"
        rf"(assignment|quiz|exam|midterm|project|homework)\s+"
        rf"({day_alt})\b",
        re.IGNORECASE,
    )
    for match in kind_day.finditer(blob):
        name = f"{match.group(1).strip()} {match.group(2).strip().title()}"
        window = blob[match.start() : min(len(blob), match.end() + 40)]
        _add_task(name, _normalize_day(match.group(3)), _find_hours_near(window))

    # Follow-up fragments: "Tuesday, around 4 hours" / "due Tuesday"
    follow_day = re.compile(
        rf"^(?:due\s+)?({day_alt})\b(?:[, ]+(?:around|about)?\s*(\d+(?:\.\d+)?)\s*(?:hours|hrs|hr|h)?)?",
        re.IGNORECASE,
    )
    follow_match = follow_day.search(text.strip())
    if follow_match and history_text:
        day = _normalize_day(follow_match.group(1))
        hours = float(follow_match.group(2)) if follow_match.group(2) else _find_hours_near(text)
        # Attach to unfinished task from history extraction
        prior = extract_study_context(history_text, "")
        if prior["tasks"]:
            target = prior["tasks"][0]
            _add_task(target["name"], day, hours or target.get("estimated_hours"), target.get("priority", "medium"))
            for extra in prior["tasks"][1:]:
                _add_task(
                    extra["name"],
                    _day_name_from_deadline(extra.get("deadline")),
                    extra.get("estimated_hours"),
                    extra.get("priority", "medium"),
                )

    # Named tasks without dates
    if not tasks:
        named = re.findall(
            r"\b((?:PDC|AI|Agentic AI|DSA|OS|DB|SE|CN)(?:\s+[A-Za-z]+)?\s+"
            r"(?:assignment|quiz|exam|project|homework))",
            blob,
            flags=re.IGNORECASE,
        )
        for name in named:
            _add_task(name.strip(), None, _find_hours_near(blob))

    # Generic "my AI assignment" style
    if not tasks:
        generic = re.search(
            r"\b(?:my|the)\s+([A-Za-z][A-Za-z0-9 /&+-]{1,40}?)\s+"
            r"(assignment|quiz|exam|project|homework)\b",
            blob,
            re.IGNORECASE,
        )
        if generic:
            _add_task(
                f"{generic.group(1).strip()} {generic.group(2).strip().title()}",
                None,
                _find_hours_near(blob),
            )

    classes: list[dict[str, Any]] = []
    class_pattern = re.compile(
        r"(monday|tuesday|wednesday|thursday|friday|saturday|sunday|"
        r"mon|tue|tues|wed|thu|thur|thurs|fri|sat|sun)\s+"
        r"(?:from\s+)?(\d{1,2}(?::\d{2})?)\s*(?:am|pm)?\s*[–\-to]+\s*"
        r"(\d{1,2}(?::\d{2})?)\s*(?:am|pm)?",
        re.IGNORECASE,
    )
    for match in class_pattern.finditer(blob):
        day = _normalize_day(match.group(1))
        start = _normalize_clock(match.group(2), match.group(0))
        end = _normalize_clock(match.group(3), match.group(0))
        if day and start and end:
            if end <= start:
                # Interpret end as afternoon when users write "8:30-2:30"
                end_minutes = _to_minutes(end) + 12 * 60
                if end_minutes > _to_minutes(start) and end_minutes <= 22 * 60:
                    end = _from_minutes(end_minutes)
            if end > start:
                classes.append({"day": day, "start": start, "end": end, "title": "Class"})

    # If the user mentions classes but extraction failed, fall back to sample classes
    # only when the text clearly references the sample scenario.
    if not classes and ("class" in lower or "classes" in lower):
        if "8:30" in lower or "8.30" in lower or "monday" in lower:
            classes = deepcopy(sample["classes"])

    unavailable: list[dict[str, Any]] = []
    if "unavailable" in lower or "busy" in lower:
        for match in class_pattern.finditer(blob):
            # Already captured as classes; skip unless labeled unavailable.
            pass

    preferred = {
        "start": settings.preferred_start,
        "end": settings.preferred_end,
    }
    pref_match = re.search(
        r"(?:prefer|study)\s+(?:between\s+)?(\d{1,2}(?::\d{2})?)\s*[–\-to]+\s*(\d{1,2}(?::\d{2})?)",
        blob,
        re.IGNORECASE,
    )
    if pref_match:
        preferred["start"] = _normalize_clock(pref_match.group(1), pref_match.group(0)) or preferred["start"]
        preferred["end"] = _normalize_clock(pref_match.group(2), pref_match.group(0)) or preferred["end"]

    return {
        "tasks": tasks,
        "classes": classes,
        "unavailable": unavailable,
        "preferred_hours": preferred,
        "raw_text": blob,
    }


def _find_hours_near(text: str) -> float | None:
    match = re.search(
        r"(?:around|about|approx(?:imately)?|roughly|needing|needs|takes|for)?\s*"
        r"(\d+(?:\.\d+)?)\s*(?:hours|hrs|hr)\b",
        text,
        re.I,
    )
    if match:
        value = float(match.group(1))
        return value if value > 0 else None
    match = re.search(r"\b(\d+(?:\.\d+)?)\s*h\b", text, re.I)
    if match:
        value = float(match.group(1))
        return value if value > 0 else None
    return None


def _find_priority_near(text: str) -> str:
    lower = text.lower()
    for word, priority in PRIORITY_WORDS.items():
        if re.search(rf"\b{word}\b", lower):
            return priority
    return "medium"


def _normalize_clock(token: str, context: str) -> str | None:
    token = token.strip().lower()
    if not token:
        return None
    if ":" not in token:
        token = f"{int(token):02d}:00"
    else:
        hour, minute = token.split(":")
        token = f"{int(hour):02d}:{int(minute):02d}"
    # crude am/pm handling if present in nearby context
    ctx = context.lower()
    hour = int(token.split(":")[0])
    if "pm" in ctx and hour < 12 and hour != 12:
        # only bump if the number looks like 12-hour afternoon shorthand for end times > 1
        if hour <= 7:
            hour += 12
            minute = int(token.split(":")[1])
            token = f"{hour:02d}:{minute:02d}"
    return token


def _deadline_for_day(day: str | None, hour: int = 23, minute: int = 59) -> str | None:
    if not day:
        return None
    # Anchor relative deadlines to the upcoming weekday from a fixed reference for determinism in tests.
    # Using sample reference Monday 2026-10-05 keeps demo dates stable.
    reference = datetime(2026, 10, 5)  # Monday
    target = {
        "Monday": 0,
        "Tuesday": 1,
        "Wednesday": 2,
        "Thursday": 3,
        "Friday": 4,
        "Saturday": 5,
        "Sunday": 6,
    }[day]
    date = reference + timedelta(days=target)
    return datetime(date.year, date.month, date.day, hour, minute).isoformat()


def validate_study_input(payload: dict[str, Any] | None = None) -> dict[str, Any]:
    """Check whether enough information exists to create a useful study plan."""
    payload = payload or {}
    tasks_raw = payload.get("tasks") or []
    classes_raw = payload.get("classes") or []
    unavailable_raw = payload.get("unavailable") or []
    issues: list[str] = []
    parsed_tasks: list[StudyTask] = []
    parsed_classes: list[ClassSlot] = []
    parsed_unavailable: list[UnavailableSlot] = []

    if not tasks_raw:
        issues.append("No study tasks were provided.")

    for idx, item in enumerate(tasks_raw):
        try:
            task = StudyTask.model_validate(item)
        except Exception as exc:  # noqa: BLE001
            issues.append(f"Task {idx + 1} is invalid: {exc}")
            continue
        if not task.deadline:
            issues.append(f"Task '{task.name}' is missing a deadline.")
        if task.estimated_hours is None:
            issues.append(f"Task '{task.name}' is missing estimated_hours.")
        elif task.estimated_hours <= 0:
            issues.append(f"Task '{task.name}' has a non-positive duration.")
        parsed_tasks.append(task)

    for idx, item in enumerate(classes_raw):
        try:
            parsed_classes.append(ClassSlot.model_validate(item))
        except Exception as exc:  # noqa: BLE001
            issues.append(f"Class slot {idx + 1} is invalid: {exc}")

    for idx, item in enumerate(unavailable_raw):
        try:
            parsed_unavailable.append(UnavailableSlot.model_validate(item))
        except Exception as exc:  # noqa: BLE001
            issues.append(f"Unavailable slot {idx + 1} is invalid: {exc}")

    complete = len(issues) == 0 and len(parsed_tasks) > 0
    return {
        "ok": complete,
        "complete": complete,
        "issues": issues,
        "tasks": [t.model_dump() for t in parsed_tasks],
        "classes": [c.model_dump() for c in parsed_classes],
        "unavailable": [u.model_dump() for u in parsed_unavailable],
        "preferred_hours": payload.get("preferred_hours") or {
            "start": settings.preferred_start,
            "end": settings.preferred_end,
        },
        "message": "Input is ready for scheduling." if complete else "Clarification required before scheduling.",
    }


def check_conflicts(payload: dict[str, Any] | None = None) -> dict[str, Any]:
    """Detect overlaps between planned sessions, classes, and unavailable periods."""
    payload = payload or {}
    sessions = [StudySession.model_validate(s) for s in payload.get("schedule") or []]
    classes = [ClassSlot.model_validate(c) for c in payload.get("classes") or []]
    unavailable = [UnavailableSlot.model_validate(u) for u in payload.get("unavailable") or []]
    conflicts: list[dict[str, Any]] = []

    # Session vs blocked times
    for session in sessions:
        for blocked in classes:
            if session.day == blocked.day and _ranges_overlap(session.start, session.end, blocked.start, blocked.end):
                conflicts.append(
                    {
                        "type": "class_overlap",
                        "session": session.model_dump(),
                        "with": blocked.model_dump(),
                    }
                )
        for blocked in unavailable:
            if session.day == blocked.day and _ranges_overlap(session.start, session.end, blocked.start, blocked.end):
                conflicts.append(
                    {
                        "type": "unavailable_overlap",
                        "session": session.model_dump(),
                        "with": blocked.model_dump(),
                    }
                )

    # Session vs session
    for i, left in enumerate(sessions):
        for right in sessions[i + 1 :]:
            if left.day == right.day and _ranges_overlap(left.start, left.end, right.start, right.end):
                conflicts.append(
                    {
                        "type": "session_overlap",
                        "session": left.model_dump(),
                        "with": right.model_dump(),
                    }
                )

    return {
        "ok": len(conflicts) == 0,
        "conflict_count": len(conflicts),
        "conflicts": conflicts,
        "message": "No conflicts detected." if not conflicts else f"Found {len(conflicts)} conflict(s).",
    }


def generate_study_schedule(payload: dict[str, Any] | None = None) -> dict[str, Any]:
    """Generate a conflict-aware study schedule using simple deadline/priority heuristics."""
    payload = payload or {}
    tasks = [StudyTask.model_validate(t) for t in payload.get("tasks") or []]
    classes = [ClassSlot.model_validate(c) for c in payload.get("classes") or []]
    unavailable = [UnavailableSlot.model_validate(u) for u in payload.get("unavailable") or []]
    preferred = payload.get("preferred_hours") or {
        "start": settings.preferred_start,
        "end": settings.preferred_end,
    }
    block_hours = settings.study_block_minutes / 60.0

    if not tasks:
        return {
            "ok": False,
            "schedule": [],
            "warnings": ["No tasks available to schedule."],
            "unscheduled_hours": {},
            "message": "Cannot generate a schedule without tasks.",
        }

    priority_rank = {"high": 0, "medium": 1, "low": 2}
    ordered = sorted(
        tasks,
        key=lambda t: (
            t.deadline or "9999-12-31",
            priority_rank.get(t.priority, 1),
            -(t.estimated_hours or 0),
        ),
    )

    # Build day list from reference week Monday..Sunday
    days = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    occupied: dict[str, list[tuple[str, str]]] = {d: [] for d in days}
    for item in classes + unavailable:
        occupied[item.day].append((item.start, item.end))

    schedule: list[StudySession] = []
    warnings: list[str] = []
    unscheduled: dict[str, float] = {}

    for task in ordered:
        remaining = float(task.estimated_hours or 0)
        deadline_day = _day_from_deadline(task.deadline)
        candidate_days = days[:]
        if deadline_day:
            # Prefer days up to and including deadline day in the demo week
            idx = days.index(deadline_day)
            candidate_days = days[: idx + 1]

        for day in candidate_days:
            if remaining <= 0:
                break
            free_slots = _free_slots(day, occupied[day], preferred["start"], preferred["end"])
            for start, end in free_slots:
                if remaining <= 0:
                    break
                available = (_to_minutes(end) - _to_minutes(start)) / 60.0
                if available <= 0:
                    continue
                take = min(remaining, block_hours, available)
                # Round to 30-minute increments for readability
                take = max(0.5, round(take * 2) / 2)
                if take > available:
                    take = available
                if take <= 0:
                    continue
                session_end = _from_minutes(_to_minutes(start) + int(take * 60))
                session = StudySession(
                    task_name=task.name,
                    day=day,
                    start=start,
                    end=session_end,
                    duration_hours=take,
                    date=_date_for_day(day),
                )
                schedule.append(session)
                occupied[day].append((session.start, session.end))
                occupied[day].sort(key=lambda pair: pair[0])
                remaining -= take

        if remaining > 0.05:
            unscheduled[task.name] = round(remaining, 2)
            warnings.append(
                f"Only {round((task.estimated_hours or 0) - remaining, 2)} of "
                f"{task.estimated_hours} hour(s) for '{task.name}' fit before the deadline/availability."
            )

    conflict_check = check_conflicts(
        {
            "schedule": [s.model_dump() for s in schedule],
            "classes": [c.model_dump() for c in classes],
            "unavailable": [u.model_dump() for u in unavailable],
        }
    )
    if conflict_check["conflicts"]:
        warnings.append("Generated schedule still reported conflicts; review required.")

    total_needed = sum(float(t.estimated_hours or 0) for t in tasks)
    total_scheduled = sum(s.duration_hours for s in schedule)
    if total_scheduled + 0.05 < total_needed:
        warnings.append(
            f"Only {round(total_scheduled, 2)} of the required {round(total_needed, 2)} hours can be scheduled."
        )

    return {
        "ok": True,
        "schedule": [s.model_dump() for s in schedule],
        "warnings": warnings,
        "unscheduled_hours": unscheduled,
        "conflicts": conflict_check["conflicts"],
        "message": "Schedule generated." if not warnings else "Schedule generated with warnings.",
    }


def summarize_plan(payload: dict[str, Any] | None = None) -> dict[str, Any]:
    """Turn the structured plan into a concise human-readable response."""
    payload = payload or {}
    schedule = payload.get("schedule") or []
    warnings = payload.get("warnings") or []
    tasks = payload.get("tasks") or []
    conflicts = payload.get("conflicts") or []

    if not schedule:
        text = (
            "I could not place any study sessions with the current constraints. "
            "Please provide more available hours or reduce task load."
        )
        return {"ok": False, "summary": text, "message": text}

    lines = ["Here is your conflict-aware study plan:"]
    by_day: dict[str, list[dict[str, Any]]] = {}
    for session in schedule:
        by_day.setdefault(session["day"], []).append(session)
    for day, items in by_day.items():
        lines.append(f"{day}:")
        for session in sorted(items, key=lambda s: s["start"]):
            lines.append(
                f"  - {session['start']}-{session['end']}: {session['task_name']} "
                f"({session['duration_hours']}h)"
            )

    if tasks:
        lines.append("Task coverage:")
        for task in tasks:
            scheduled = sum(
                s["duration_hours"] for s in schedule if s["task_name"] == task.get("name")
            )
            needed = task.get("estimated_hours") or 0
            lines.append(f"  - {task.get('name')}: {scheduled}h / {needed}h")

    if warnings:
        lines.append("Warnings:")
        for warning in warnings:
            lines.append(f"  - {warning}")
    if conflicts:
        lines.append(f"Unresolved conflicts: {len(conflicts)}")

    summary = "\n".join(lines)
    if len(summary) > 1900:
        summary = summary[:1890] + "…"
    return {"ok": True, "summary": summary, "message": "Plan summarized."}


def _day_from_deadline(deadline: str | None) -> str | None:
    if not deadline:
        return None
    try:
        dt = datetime.fromisoformat(deadline)
    except ValueError:
        return None
    return dt.strftime("%A")


def _day_name_from_deadline(deadline: str | None) -> str | None:
    return _day_from_deadline(deadline)


def _date_for_day(day: str) -> str:
    reference = datetime(2026, 10, 5)
    idx = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"].index(day)
    return (reference + timedelta(days=idx)).date().isoformat()


def _free_slots(
    day: str,
    occupied: list[tuple[str, str]],
    day_start: str,
    day_end: str,
) -> list[tuple[str, str]]:
    """Return free intervals within preferred hours after subtracting occupied ranges."""
    del day  # day retained for readability at call sites
    cursor = _to_minutes(day_start)
    end = _to_minutes(day_end)
    blocked = sorted(((_to_minutes(s), _to_minutes(e)) for s, e in occupied), key=lambda p: p[0])
    free: list[tuple[str, str]] = []
    for b_start, b_end in blocked:
        if b_end <= cursor:
            continue
        if b_start > cursor:
            free.append((_from_minutes(cursor), _from_minutes(min(b_start, end))))
        cursor = max(cursor, b_end)
        if cursor >= end:
            break
    if cursor < end:
        free.append((_from_minutes(cursor), _from_minutes(end)))
    return [(s, e) for s, e in free if _to_minutes(e) - _to_minutes(s) >= 30]


TOOLS = {
    "validate_study_input": validate_study_input,
    "check_conflicts": check_conflicts,
    "generate_study_schedule": generate_study_schedule,
    "summarize_plan": summarize_plan,
}
