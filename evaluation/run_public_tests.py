"""Sends HTTP requests against a running StudyPlan Agent server."""
import argparse
import json
from pathlib import Path

import httpx

parser = argparse.ArgumentParser()
parser.add_argument("--url", default="http://127.0.0.1:8000")
args = parser.parse_args()
cases = json.loads(Path(__file__).with_name("public_cases.json").read_text(encoding="utf-8"))
passed = 0
for case in cases:
    try:
        response = httpx.post(args.url.rstrip("/") + "/arena/run", json=case["request"], timeout=70)
        response.raise_for_status()
        result = response.json()
        ok = result["status"] == case["expected_status"] and result["stop_reason"] == case["expected_stop_reason"]
        if ok:
            passed += 1
        print(("PASS" if ok else "FAIL") + " " + case["name"] + f" -> {result.get('status')}/{result.get('stop_reason')}")
    except (httpx.HTTPError, KeyError, ValueError) as exc:
        print("FAIL " + case["name"] + " " + type(exc).__name__ + " " + str(exc))
print(f"{passed}/{len(cases)} checks passed.")
raise SystemExit(0 if passed == len(cases) else 1)
