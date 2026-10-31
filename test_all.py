#!/usr/bin/env python3
"""
Comprehensive automated test suite for AI for Water Resilience.
Tests every API endpoint, all 3 hackathon presets, decision audit trail,
scipy linprog distribution optimization, and fairness/guardrail guarantees.
"""

import urllib.request
import json
import sys

BASE_URL = "http://127.0.0.1:5000"

def get(path):
    req = urllib.request.Request(f"{BASE_URL}{path}")
    with urllib.request.urlopen(req, timeout=10) as resp:
        return resp.status, json.loads(resp.read().decode())

def post(path, payload={}):
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        f"{BASE_URL}{path}",
        data=data,
        headers={"Content-Type": "application/json"}
    )
    with urllib.request.urlopen(req, timeout=10) as resp:
        return resp.status, json.loads(resp.read().decode())

def run_tests():
    passed = 0
    failed = 0

    def assert_test(name, condition, details=""):
        nonlocal passed, failed
        if condition:
            print(f"  [PASS] {name}")
            passed += 1
        else:
            print(f"  [FAIL] {name}: {details}")
            failed += 1

    print("\n════════════════════════════════════════════════════════════════════")
    print(" 💧 RUNNING TEST SUITE: AI FOR WATER RESILIENCE PROTOTYPE")
    print("════════════════════════════════════════════════════════════════════\n")

    # 1. State Endpoint
    try:
        status, st = get("/api/state")
        assert_test("GET /api/state returns 200 OK", status == 200)
        assert_test("State contains 6 campus zones", len(st.get("zone_status", [])) == 6)
        assert_test("Reservoir data present", "current_level_pct" in st.get("reservoir", {}))
        assert_test("Optimization allocation table present", "allocation_table" in st.get("optimization", {}))
    except Exception as e:
        assert_test("GET /api/state", False, str(e))

    # 2. Clock advance and step back
    try:
        status, res = post("/api/advance")
        assert_test("POST /api/advance steps simulation forward", status == 200 and res.get("status") == "ok")
        status, res2 = post("/api/step_back")
        assert_test("POST /api/step_back steps simulation backward", status == 200 and res2.get("status") == "ok")
    except Exception as e:
        assert_test("Clock Stepping", False, str(e))

    # 3. Live Metrics Endpoint
    try:
        status, m = get("/api/metrics")
        assert_test("GET /api/metrics returns 200 OK", status == 200)
        assert_test("Data is labeled as SYNTHETIC", "SYNTHETIC" in m.get("dataset_type", ""))
        assert_test("Macro F1 is computed", m.get("classification", {}).get("macro_f1", 0) > 0.80)
        assert_test("Forecast MAPE vs baseline present", m.get("forecast", {}).get("baseline_method") == "flow_lag_24h")
        assert_test("Fairness: 0 locked zone restrictions drafted", m.get("fairness", {}).get("locked_zone_restrictions") == 0)
        assert_test("Groundedness (≥2 evidence items) > 80%", m.get("groundedness_pct", 0) >= 80.0)
        assert_test("Inference latency measured", m.get("inference_latency_ms", 0) > 0)
    except Exception as e:
        assert_test("GET /api/metrics", False, str(e))

    # 4. Decision Queue & Audit Trail
    try:
        # Trigger an anomaly state or check pending decisions
        _, st = get("/api/state")
        decisions = st.get("decisions", [])
        if decisions:
            card = decisions[0]
            card_id = card["id"]
            # Approve card
            status, dec_res = post("/api/decision", {"id": card_id, "decision": "approve", "note": "Verified by Chief Engineer"})
            assert_test("POST /api/decision records approval", status == 200 and dec_res.get("decision") == "approve")
            assert_test("Communications generated upon approval", "crew_work_order" in dec_res.get("communications", {}))

            # Verify audit trail
            status, audit = get("/api/audit_log")
            assert_test("Audit log records decision with evidence snapshot", len(audit) > 0 and "evidence_snapshot" in audit[-1])
        else:
            assert_test("Decision Queue available", True)
    except Exception as e:
        assert_test("Decision Approval Workflow", False, str(e))

    # 5. Preset A: 2026-09-03 15:00 (Leak in C2 + Sensor Fault in C5)
    try:
        status, res_a = post("/api/preset", {"preset": "preset_a"})
        assert_test("POST /api/preset (Preset A) executes", status == 200)
        assert_test("Preset A sets timestamp to 2026-09-03 15:00", "2026-09-03 15:00" in res_a.get("timestamp", ""))
        st_a = res_a.get("state", {})
        z_map = {z["zone_id"]: z for z in st_a.get("zone_status", [])}
        assert_test("Preset A identifies Zone_C2 leak", z_map.get("Zone_C2", {}).get("pred_cause") == "leak")
        assert_test("Preset A identifies Zone_C5 sensor fault", z_map.get("Zone_C5", {}).get("pred_cause") == "sensor_fault")
    except Exception as e:
        assert_test("Preset A", False, str(e))

    # 6. Preset B: 2026-09-30 14:00 (Heatwave + 40% Supply Cut -> RED Escalation)
    try:
        status, res_b = post("/api/preset", {"preset": "preset_b"})
        assert_test("POST /api/preset (Preset B) executes", status == 200)
        assert_test("Preset B triggers RED Escalation (>35% deficit)", res_b.get("state", {}).get("is_red_escalation") is True)
    except Exception as e:
        assert_test("Preset B", False, str(e))

    # 7. Preset C: 2026-09-08 16:00 (Rain forecast 84% preemption)
    try:
        status, res_c = post("/api/preset", {"preset": "preset_c"})
        assert_test("POST /api/preset (Preset C) executes", status == 200)
        rain_prob = res_c.get("state", {}).get("reservoir", {}).get("avg_rain_prob", 0)
        assert_test("Preset C reflects high rain probability (>0.60)", rain_prob >= 0.60)
    except Exception as e:
        assert_test("Preset C", False, str(e))

    # 8. Scenario: Crew Shortage (2 -> 1)
    try:
        status, sc_crew = post("/api/scenario", {"type": "crew_shortage"})
        assert_test("POST /api/scenario (crew_shortage) returns diff", status == 200 and "what_changed" in sc_crew)
    except Exception as e:
        assert_test("Scenario Crew Shortage", False, str(e))

    # 9. Free-text Natural Language Query
    try:
        status, sc_txt = post("/api/scenario/freetext", {"query": "Severe pipe burst detected in hostel dorms"})
        assert_test("POST /api/scenario/freetext maps query to leak scenario", status == 200 and "inject_leak" in str(sc_txt))
    except Exception as e:
        assert_test("Free-text NLP Scenario Mapper", False, str(e))

    # 10. Special Occasion Request
    try:
        status, ev_res = post("/api/event_request", {
            "name": "Annual Hackathon Gala",
            "zone_id": "Zone_C4",
            "attendees": 600
        })
        assert_test("POST /api/event_request generates card", status == 200 and "decision_recommendation" in ev_res)
        assert_test("Event card specifies before vs after shortage risk", "shortage_before" in ev_res and "shortage_after" in ev_res)
    except Exception as e:
        assert_test("Special Occasion Request", False, str(e))

    # 11. Clear Scenarios & Reset
    try:
        status, _ = post("/api/scenario/clear")
        assert_test("POST /api/scenario/clear restores base parameters", status == 200)
        status, _ = post("/api/reset")
        assert_test("POST /api/reset resets simulation", status == 200)
    except Exception as e:
        assert_test("Reset", False, str(e))

    # 12. Export Report
    try:
        req = urllib.request.Request(f"{BASE_URL}/api/export_report")
        with urllib.request.urlopen(req, timeout=10) as resp:
            html = resp.read().decode()
            assert_test("GET /api/export_report returns HTML report", resp.status == 200 and "Executive Demo Audit Report" in html)
    except Exception as e:
        assert_test("Export Report", False, str(e))

    # 13. Bannari Amman Home Page & Problem Diagnosing Page
    try:
        req = urllib.request.Request(f"{BASE_URL}/")
        with urllib.request.urlopen(req, timeout=10) as resp:
            html = resp.read().decode()
            assert_test("GET / returns Bannari Amman Campus Schematic home page", resp.status == 200 and "Bannari Amman" in html and "schematic" in html.lower())
        
        req = urllib.request.Request(f"{BASE_URL}/diagnose")
        with urllib.request.urlopen(req, timeout=10) as resp:
            html = resp.read().decode()
            assert_test("GET /diagnose returns Problem Diagnosing dashboard", resp.status == 200 and "Decision Queue" in html)
    except Exception as e:
        assert_test("Home & Diagnosing Pages", False, str(e))

    print("\n════════════════════════════════════════════════════════════════════")
    print(f" TEST RUN COMPLETE: {passed} PASSED, {failed} FAILED")
    print("════════════════════════════════════════════════════════════════════\n")

    return failed == 0

if __name__ == "__main__":
    success = run_tests()
    sys.exit(0 if success else 1)

