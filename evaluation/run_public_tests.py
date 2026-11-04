"""Check a running StudyPlan Agent over HTTP: endpoints, public cases, and a two-turn clarification.

  .\\.venv\\Scripts\\python.exe evaluation\\run_public_tests.py --url https://YOUR-APP.onrender.com --out evaluation\\public_results.json
"""
import argparse
import json
import time
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

import httpx

parser = argparse.ArgumentParser()
parser.add_argument("--url", default="http://127.0.0.1:8000")
parser.add_argument("--out", help="Write machine-readable results to this JSON file")
parser.add_argument("--pause", type=float, default=0, help="Seconds between cases, to respect free-tier limits")
args = parser.parse_args()
base = args.url.rstrip("/")
cases = json.loads(Path(__file__).with_name("public_cases.json").read_text(encoding="utf-8"))
results = []


def record(name, ok, detail):
    results.append({"name": name, "passed": ok, **detail})
    print(("PASS " if ok else "FAIL ") + name + " -> " + json.dumps({k: detail.get(k) for k in ("status", "stop_reason", "steps", "latency_ms") if k in detail}))


# Cold starts on free hosting can take about a minute.
health = httpx.get(base + "/health", timeout=120)
health_body = health.json()
record(
    "GET /health reports a live model",
    health.status_code == 200 and health_body.get("live_llm") is True,
    {"status": health_body.get("status"), "active_model": health_body.get("active_model")},
)
manifest = httpx.get(base + "/arena/manifest", timeout=60)
record("GET /arena/manifest", manifest.status_code == 200 and manifest.json().get("arena_version") == "0.1", {"status": str(manifest.status_code)})

for case in cases:
    time.sleep(args.pause)
    started = time.perf_counter()
    try:
        response = httpx.post(base + "/arena/run", json=case["request"], timeout=70)
        response.raise_for_status()
        result = response.json()
        ok = result["status"] == case["expected_status"] and result["stop_reason"] == case["expected_stop_reason"]
        record(
            case["name"],
            ok,
            {
                "status": result["status"],
                "stop_reason": result["stop_reason"],
                "steps": result["steps"],
                "tools": [t["tool"] for t in result["tool_calls"]],
                "metrics": result["metrics"],
                "latency_ms": round((time.perf_counter() - started) * 1000),
                "final_response": result["final_response"],
            },
        )
    except (httpx.HTTPError, KeyError, ValueError) as exc:
        record(case["name"], False, {"error": type(exc).__name__ + " " + str(exc)})

# One clarification across two messages in the same chat session.
time.sleep(args.pause)
session = "public-test-" + uuid4().hex
turns = []
try:
    for text in ("Make me a plan for my AI assignment.", "It's due Friday and needs about 4 hours."):
        reply = httpx.post(base + "/chat", json={"session_id": session, "task": text}, timeout=70)
        reply.raise_for_status()
        body = reply.json()
        turns.append({"user": text, "status": body["status"], "agent": body["final_response"]})
    ok = [t["status"] for t in turns] == ["needs_clarification", "completed"] and "AI" in turns[1]["agent"]
    record("Multi-turn clarification keeps the goal", ok, {"status": turns[-1]["status"], "turns": turns})
except (httpx.HTTPError, KeyError, ValueError) as exc:
    record("Multi-turn clarification keeps the goal", False, {"error": type(exc).__name__ + " " + str(exc)})
finally:
    httpx.delete(base + "/chat/" + session, timeout=30)

passed = sum(r["passed"] for r in results)
print(f"{passed}/{len(results)} checks passed.")
if args.out:
    Path(args.out).write_text(
        json.dumps(
            {
                "target": base,
                "tested_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                "active_model": health_body.get("active_model"),
                "passed": passed,
                "total": len(results),
                "results": results,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
raise SystemExit(0 if passed == len(results) else 1)
