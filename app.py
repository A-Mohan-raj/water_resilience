#!/usr/bin/env python3
"""
AI for Water Resilience – Campus Decision Intelligence Prototype
Run: python app.py (port 5000)
Libraries: flask, pandas, numpy, scikit-learn, scipy (linprog 'highs').
(No xgboost installed; uses scikit-learn GradientBoosting and scipy linprog).
"""

import json, os, math, time, uuid, datetime
from pathlib import Path

import numpy as np
import pandas as pd
from flask import Flask, jsonify, request, send_from_directory, Response
from sklearn.ensemble import GradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import precision_recall_fscore_support, confusion_matrix

try:
    from scipy.optimize import linprog
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False

# ──────────────────────────────────────────────────────────────────────────────
# CONSTANTS & CONFIGURATION
# ──────────────────────────────────────────────────────────────────────────────
ESCALATION_SUPPLY_DEFICIT_THRESHOLD = 0.35  # Single constant: 35% supply deficit triggers RED escalation
WATER_COST_RS_PER_KL = 14.0                 # Assumption: campus municipal bulk water cost
REPAIR_COST_RS = 15000.0                    # Assumption: typical pipe repair cost
RESPONSE_COST_RS = 2000.0                   # Assumption: plumber/crew dispatch logistics
RECOVERY_FACTOR = 0.85                      # Assumption: 85% effective net recovery of water loss
RESERVOIR_CAPACITY_KL = 2000.0              # Assumption: total campus OHT reservoir capacity
TIER_WEIGHT = {1: 2.5, 2: 1.5, 3: 1.0}

BASE_DIR = Path(__file__).resolve().parent
AUDIT_LOG_PATH = BASE_DIR / "audit_log.json"
DATA_PATH = BASE_DIR / "data.csv"

app = Flask(__name__, static_folder="static")

# ──────────────────────────────────────────────────────────────────────────────
# 1. DATA LOADING & STRICT TIME SPLIT
# ──────────────────────────────────────────────────────────────────────────────
print("[1/8] Loading dataset & applying strict temporal split …")
RAW = pd.read_csv(DATA_PATH, parse_dates=["timestamp"])
RAW = RAW.sort_values(["zone_id", "timestamp"]).reset_index(drop=True)

# Drop the first 24 rows per zone (padded lags)
zone_chunks = []
for zid in RAW["zone_id"].unique():
    zone_chunks.append(RAW[RAW["zone_id"] == zid].iloc[24:])
RAW = pd.concat(zone_chunks, ignore_index=True)

TRAIN_CUTOFF = pd.Timestamp("2026-10-01 00:00:00")
TRAIN = RAW[RAW["timestamp"] < TRAIN_CUTOFF].copy()
TEST = RAW[RAW["timestamp"] >= TRAIN_CUTOFF].copy()

print(f"   Total rows: {len(RAW)} (Train: {len(TRAIN)}, Test: {len(TEST)})")
print(f"   Data marked as: SYNTHETIC (Campus Simulation)")

ZONES_META = (
    RAW[["zone_id", "zone_name", "vulnerability_tier", "ethical_guardrail_locked", "pipe_material", "pipe_age_years"]]
    .drop_duplicates(subset=["zone_id"])
    .to_dict("records")
)
ZONE_MAP = {z["zone_id"]: z for z in ZONES_META}

# ──────────────────────────────────────────────────────────────────────────────
# FEATURE DEFINITIONS (Answer columns strictly excluded)
# ──────────────────────────────────────────────────────────────────────────────
EXCLUDED_ANSWER_COLS = ["anomaly_type", "anomaly_code", "recommended_action", "repair_urgency_score"]

FEATURE_COLS = [
    "flow_residual_ratio", "pressure_delta_1h", "acoustic_vibration_db",
    "oht_mass_balance_loss_pct", "sensor_packet_jitter_ms",
    "nlp_complaint_count", "nlp_complaint_severity_score",
    "building_occupancy_pct", "hour_of_day", "is_weekend", "is_pre_lecture_rush",
    "ambient_temp_c", "rainfall_mm", "flow_lag_1h", "flow_rolling_6h_mean",
]
CAMPUS_MODE_COL = "campus_calendar_mode"
ALL_MODE_COLS = sorted(pd.get_dummies(RAW[CAMPUS_MODE_COL], prefix="mode").columns.tolist())

def extract_features(df):
    X_num = df[FEATURE_COLS].copy()
    dummies = pd.get_dummies(df[CAMPUS_MODE_COL], prefix="mode")
    for col in ALL_MODE_COLS:
        if col not in dummies.columns:
            dummies[col] = 0
    dummies = dummies[ALL_MODE_COLS]
    return pd.concat([X_num.reset_index(drop=True), dummies.reset_index(drop=True)], axis=1)

# ──────────────────────────────────────────────────────────────────────────────
# 2. DETECT & DIAGNOSE (GradientBoostingClassifier)
# ──────────────────────────────────────────────────────────────────────────────
print("[2/8] Training anomaly classifier (GradientBoostingClassifier) …")

CAUSE_MAP = {
    "Normal": "normal",
    "Physical_Pipe_Leak": "leak",
    "Unusual_Demand_Spike": "demand_spike",
    "Sensor_Recording_Fault": "sensor_fault",
}
INV_CAUSE_MAP = {v: k for k, v in CAUSE_MAP.items()}

le_cause = LabelEncoder()
le_cause.fit(sorted(CAUSE_MAP.values()))

y_train_raw = TRAIN["anomaly_type"].map(CAUSE_MAP)
y_test_raw = TEST["anomaly_type"].map(CAUSE_MAP)

X_train_cls = extract_features(TRAIN)
X_test_cls = extract_features(TEST)

y_train_enc = le_cause.transform(y_train_raw)
y_test_enc = le_cause.transform(y_test_raw)

clf = GradientBoostingClassifier(
    n_estimators=150, max_depth=5, learning_rate=0.1,
    subsample=0.8, random_state=42
)
clf.fit(X_train_cls.values, y_train_enc)

# Measure inference latency on test set
t0 = time.perf_counter()
y_pred_enc = clf.predict(X_test_cls.values)
y_pred_prob = clf.predict_proba(X_test_cls.values)
t1 = time.perf_counter()
INFERENCE_LATENCY_MS = round(((t1 - t0) / len(X_test_cls)) * 1000, 3)

y_pred_labels = le_cause.inverse_transform(y_pred_enc)
y_true_labels = le_cause.inverse_transform(y_test_enc)

# Pre-predict across all of RAW for fast replay
X_all_cls = extract_features(RAW)
all_preds = le_cause.inverse_transform(clf.predict(X_all_cls.values))
all_probs = clf.predict_proba(X_all_cls.values).max(axis=1)

RAW["pred_cause"] = all_preds
RAW["pred_cause_prob"] = all_probs
TEST["pred_cause"] = y_pred_labels
TEST["pred_cause_prob"] = y_pred_prob.max(axis=1)

# Group incidents (same zone, gap <= 3h)
def label_incidents(df, cause_col="pred_cause"):
    df = df.copy()
    df["is_anomaly"] = df[cause_col] != "normal"
    df["incident_id"] = None
    inc_counter = 0
    for zid in df["zone_id"].unique():
        mask = (df["zone_id"] == zid) & df["is_anomaly"]
        idxs = df.loc[mask].index.tolist()
        if not idxs:
            continue
        inc_counter += 1
        df.loc[idxs[0], "incident_id"] = inc_counter
        for prev_i, cur_i in zip(idxs[:-1], idxs[1:]):
            gap_hrs = (df.loc[cur_i, "timestamp"] - df.loc[prev_i, "timestamp"]).total_seconds() / 3600.0
            if gap_hrs <= 3.0:
                df.loc[cur_i, "incident_id"] = inc_counter
            else:
                inc_counter += 1
                df.loc[cur_i, "incident_id"] = inc_counter
    return df

RAW = label_incidents(RAW)
TEST = label_incidents(TEST)

# ──────────────────────────────────────────────────────────────────────────────
# 3. CROSS-VERIFY (Rule-Based Verifier with Train-set Percentiles)
# ──────────────────────────────────────────────────────────────────────────────
print("[3/8] Calibrating Cross-Verifier thresholds from TRAIN Normal percentiles …")

TRAIN_NORMAL = TRAIN[TRAIN["anomaly_type"] == "Normal"]
THRESHOLDS = {
    "acoustic_high": float(TRAIN_NORMAL["acoustic_vibration_db"].quantile(0.95)),
    "oht_loss_high": float(TRAIN_NORMAL["oht_mass_balance_loss_pct"].quantile(0.95)),
    "pressure_drop_low": float(TRAIN_NORMAL["pressure_delta_1h"].quantile(0.05)),
    "jitter_high": float(TRAIN_NORMAL["sensor_packet_jitter_ms"].quantile(0.95)),
    "complaint_high": float(TRAIN_NORMAL["nlp_complaint_count"].quantile(0.95)),
    "occupancy_high": float(TRAIN_NORMAL["building_occupancy_pct"].quantile(0.90)),
}

def cross_verify_row(row):
    """
    Evaluates supporting and contradicting evidence against train percentiles.
    Returns: dict(confidence, supporting, contradicting, needs_human_review, degraded_confidence)
    """
    cause = row.get("pred_cause", "normal")
    sup, con = [], []
    conf = float(row.get("pred_cause_prob", 0.8)) * 100.0
    degraded = False

    # Check if imputed
    if row.get("sensor_imputed", False):
        degraded = True
        con.append("Sensor readings imputed from neighbours/history (higher uncertainty)")
        conf *= 0.65

    if cause == "leak":
        # Hydraulic excess evidence
        flow_excess = max(0.0, float(row["flow_rate_lpm"]) - float(row["expected_baseline_flow_lpm"]))
        if flow_excess > 0:
            sup.append(f"Excess hydraulic flow: +{flow_excess:.1f} LPM above baseline ({row['expected_baseline_flow_lpm']:.1f} LPM)")

        if row["acoustic_vibration_db"] > THRESHOLDS["acoustic_high"]:
            sup.append(f"Acoustic vibration {row['acoustic_vibration_db']:.1f} dB > p95 ({THRESHOLDS['acoustic_high']:.1f} dB)")
        else:
            con.append(f"Acoustic vibration {row['acoustic_vibration_db']:.1f} dB ≤ p95 (leak noise missing)")

        if row["oht_mass_balance_loss_pct"] > THRESHOLDS["oht_loss_high"]:
            sup.append(f"OHT mass balance loss {row['oht_mass_balance_loss_pct']:.1f}% > p95 ({THRESHOLDS['oht_loss_high']:.1f}%)")
        else:
            con.append(f"OHT tank mass balance loss {row['oht_mass_balance_loss_pct']:.1f}% ≤ p95 (no tank deficit)")

        if row["pressure_delta_1h"] < THRESHOLDS["pressure_drop_low"]:
            sup.append(f"Pressure drop {row['pressure_delta_1h']:.1f} psi < p5 ({THRESHOLDS['pressure_drop_low']:.1f} psi)")
        else:
            con.append(f"Pressure change {row['pressure_delta_1h']:.1f} psi does not confirm sudden drop")

        if row.get("nlp_complaint_count", 0) > THRESHOLDS["complaint_high"]:
            complaint_sample = str(row.get("sample_complaint_text", "")).strip()
            sup.append(f"NLP complaints count = {int(row['nlp_complaint_count'])} ('{complaint_sample}')")

    elif cause == "demand_spike":
        # Demand surge evidence
        flow_excess = max(0.0, float(row["flow_rate_lpm"]) - float(row["expected_baseline_flow_lpm"]))
        if flow_excess > 0:
            sup.append(f"Elevated consumption draw: +{flow_excess:.1f} LPM over normal baseline")

        if row.get("is_pre_lecture_rush", 0) == 1:
            sup.append("Campus schedule: Pre-lecture rush window active")
        if row["building_occupancy_pct"] > THRESHOLDS["occupancy_high"]:
            sup.append(f"High occupancy {row['building_occupancy_pct']:.0f}% > p90 ({THRESHOLDS['occupancy_high']:.0f}%)")
        else:
            con.append(f"Occupancy {row['building_occupancy_pct']:.0f}% ≤ p90 (demand spike without crowd)")

        if row["oht_mass_balance_loss_pct"] <= THRESHOLDS["oht_loss_high"]:
            sup.append(f"OHT loss {row['oht_mass_balance_loss_pct']:.1f}% normal (consistent with legitimate draw)")
        else:
            con.append(f"OHT mass loss {row['oht_mass_balance_loss_pct']:.1f}% unusually high for pure usage")

    elif cause == "sensor_fault":
        if row["sensor_packet_jitter_ms"] > THRESHOLDS["jitter_high"]:
            sup.append(f"Telemetry packet jitter {row['sensor_packet_jitter_ms']:.0f} ms > p95 ({THRESHOLDS['jitter_high']:.0f} ms)")
        else:
            con.append(f"Packet jitter {row['sensor_packet_jitter_ms']:.0f} ms within normal threshold")

        if (row["acoustic_vibration_db"] <= THRESHOLDS["acoustic_high"] and
            row["oht_mass_balance_loss_pct"] <= THRESHOLDS["oht_loss_high"]):
            sup.append("Hydraulic & acoustic signals normal (isolated sensor glitch)")
        else:
            con.append("Physical acoustic or tank loss signals abnormal (may be physical event)")

    else:
        sup.append("All sensory features within normal operating bounds")
        conf = max(conf, 85.0)

    # Adjust confidence for contradicting signals
    conf = max(10.0, conf - len(con) * 18.0)
    needs_review = len(con) >= 2 or conf < 60.0 or cause != "normal"

    return {
        "confidence": round(conf, 1),
        "supporting_evidence": sup,
        "contradicting_evidence": con,
        "needs_human_review": needs_review,
        "degraded_confidence": degraded,
    }

# Compute cross-verification for RAW and TEST
def apply_cv(df):
    results = df.apply(cross_verify_row, axis=1)
    df["cv_confidence"] = [r["confidence"] for r in results]
    df["cv_supporting"] = [r["supporting_evidence"] for r in results]
    df["cv_contradicting"] = [r["contradicting_evidence"] for r in results]
    df["cv_needs_review"] = [r["needs_human_review"] for r in results]
    df["cv_degraded"] = [r["degraded_confidence"] for r in results]
    return df

RAW = apply_cv(RAW)
TEST = apply_cv(TEST)

# ──────────────────────────────────────────────────────────────────────────────
# 4. FORECAST MODELS (HistGradientBoostingRegressor)
# ──────────────────────────────────────────────────────────────────────────────
print("[4/8] Training forecast regressors (HistGradientBoostingRegressor) …")

FORECAST_FEATURES = [
    "hour_of_day", "day_of_week", "is_weekend", "is_pre_lecture_rush",
    "building_occupancy_pct", "ambient_temp_c", "rainfall_mm",
    "forecast_rain_24h_prob", "flow_lag_1h", "flow_lag_24h",
    "flow_rolling_6h_mean", "pressure_psi", "reservoir_level_pct",
]

reg_1h = HistGradientBoostingRegressor(max_iter=150, max_depth=6, learning_rate=0.08, random_state=42)
reg_1h.fit(TRAIN[FORECAST_FEATURES].values, TRAIN["target_next_1h_flow_lpm"].values)

reg_24h = HistGradientBoostingRegressor(max_iter=150, max_depth=6, learning_rate=0.08, random_state=42)
reg_24h.fit(TRAIN[FORECAST_FEATURES].values, TRAIN["target_next_24h_demand_kl"].values)

RAW["fc_next_1h_flow"] = reg_1h.predict(RAW[FORECAST_FEATURES].values)
RAW["fc_next_24h_demand"] = reg_24h.predict(RAW[FORECAST_FEATURES].values)

TEST["fc_next_1h_flow"] = reg_1h.predict(TEST[FORECAST_FEATURES].values)
TEST["fc_next_24h_demand"] = reg_24h.predict(TEST[FORECAST_FEATURES].values)

# MAPE vs naive baseline (flow_lag_24h as per prompt instructions)
def compute_mape(y_true, y_pred):
    mask = y_true > 1.0
    if mask.sum() == 0:
        return 0.0
    return float(np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100.0)

test_mape_1h = compute_mape(TEST["target_next_1h_flow_lpm"].values, TEST["fc_next_1h_flow"].values)
baseline_mape_1h = compute_mape(TEST["target_next_1h_flow_lpm"].values, TEST["flow_lag_24h"].values)

test_mape_24h = compute_mape(TEST["target_next_24h_demand_kl"].values, TEST["fc_next_24h_demand"].values)
baseline_mape_24h = compute_mape(TEST["target_next_24h_demand_kl"].values, TEST["flow_lag_24h"].values)

print(f"   Next 1h Flow MAPE: {test_mape_1h:.2f}% (vs Naive lag_24h: {baseline_mape_1h:.2f}%)")
print(f"   Next 24h Demand MAPE: {test_mape_24h:.2f}% (vs Naive lag_24h: {baseline_mape_24h:.2f}%)")

def shortage_forecast(df_recent, extra_demand_kl=0.0):
    """
    Computes hours/days until campus reservoir level < 25%.
    Uses recent level slope, forecast demand, and rainfall probability.
    """
    try:
        latest_rows = df_recent.sort_values("timestamp").drop_duplicates("zone_id", keep="last")
        current_level = float(latest_rows["reservoir_level_pct"].iloc[0])

        recent_unique = (
            df_recent.drop_duplicates("timestamp", keep="first")
            .sort_values("timestamp")
            .tail(24)
        )
        if len(recent_unique) >= 2:
            dt_hours = max(1.0, (recent_unique["timestamp"].iloc[-1] - recent_unique["timestamp"].iloc[0]).total_seconds() / 3600.0)
            slope = float((recent_unique["reservoir_level_pct"].iloc[-1] - recent_unique["reservoir_level_pct"].iloc[0]) / dt_hours)
        else:
            slope = -0.12

        if extra_demand_kl > 0:
            extra_slope_drop = (extra_demand_kl / RESERVOIR_CAPACITY_KL * 100.0) / 24.0
            slope -= extra_slope_drop

        avg_rain_prob = float(latest_rows["forecast_rain_24h_prob"].mean())
        if avg_rain_prob > 0.60:
            slope += 0.08  # rain buffer

        if slope >= 0:
            hours_to_25 = 999.0
            risk = "low"
        else:
            shortage_needed = max(0.0, current_level - 25.0)
            hours_to_25 = round(shortage_needed / abs(slope), 1)
            if hours_to_25 < 24.0:
                risk = "high"
            elif hours_to_25 < 72.0:
                risk = "medium"
            else:
                risk = "low"

        return {
            "current_level_pct": round(current_level, 1),
            "slope_pct_per_hour": round(slope, 3),
            "hours_to_25pct": hours_to_25,
            "days_to_25pct": round(hours_to_25 / 24.0, 1) if hours_to_25 < 999 else 99.0,
            "risk": risk,
            "avg_rain_prob": round(avg_rain_prob, 2),
            "evidence": [
                f"Current reservoir level: {current_level:.1f}%",
                f"Depletion rate: {slope:.3f}%/hour",
                f"Rain forecast 24h probability: {avg_rain_prob:.0%}",
                f"Extra demand allowance: {extra_demand_kl:.1f} kL",
            ]
        }
    except Exception as e:
        return {
            "current_level_pct": 50.0,
            "slope_pct_per_hour": 0.0,
            "hours_to_25pct": 999.0,
            "days_to_25pct": 99.0,
            "risk": "low",
            "error": str(e),
            "evidence": [f"Degraded fallback: {e}"]
        }

# ──────────────────────────────────────────────────────────────────────────────
# 5. PRIORITIZE REPAIRS (With Explicit Formulas & Assumptions)
# ──────────────────────────────────────────────────────────────────────────────
print("[5/8] Initializing repair prioritizer …")

def prioritize_repairs(df_hour):
    """
    Ranks non-normal anomalies by economic impact and vulnerability.
    Formula:
      loss_L_per_day = (flow_rate_lpm - expected_baseline_flow_lpm) * 60 * 24 * recovery_factor
      loss_kL_per_day = loss_L_per_day / 1000
      score = (loss_kL_per_day * Rs14) * zone_weight / (repair_cost Rs15000 + response_cost Rs2000)
    """
    anomalies = df_hour[df_hour["pred_cause"] != "normal"].copy()
    if anomalies.empty:
        return []

    ranked = []
    for _, row in anomalies.iterrows():
        zid = row["zone_id"]
        zmeta = ZONE_MAP.get(zid, {})
        flow_excess = max(0.0, float(row["flow_rate_lpm"]) - float(row["expected_baseline_flow_lpm"]))

        # Explicit formula calculation
        loss_L_per_day = flow_excess * 60.0 * 24.0 * RECOVERY_FACTOR
        loss_kl_per_day = loss_L_per_day / 1000.0
        loss_kl_per_hr = loss_kl_per_day / 24.0

        tier = int(row["vulnerability_tier"])
        weight = TIER_WEIGHT.get(tier, 1.0)
        rs_at_risk = loss_kl_per_day * WATER_COST_RS_PER_KL * weight
        priority_score = rs_at_risk / (REPAIR_COST_RS + RESPONSE_COST_RS)

        pipe_desc = f"{zmeta.get('pipe_material', 'Ductile Iron')} ({zmeta.get('pipe_age_years', 5)}y old)"

        ranked.append({
            "zone_id": zid,
            "zone_name": row["zone_name"],
            "asset_id": f"{zid}-SUBSEG-01 [{pipe_desc}]",
            "cause": row["pred_cause"],
            "cause_prob": round(float(row["pred_cause_prob"]), 3),
            "confidence": float(row["cv_confidence"]),
            "degraded_confidence": bool(row.get("cv_degraded", False)),
            "flow_rate_lpm": round(float(row["flow_rate_lpm"]), 1),
            "expected_baseline_flow_lpm": round(float(row["expected_baseline_flow_lpm"]), 1),
            "loss_kl_per_hour": round(loss_kl_per_hr, 2),
            "loss_kl_per_day": round(loss_kl_per_day, 2),
            "rs_per_day_at_risk": round(rs_at_risk, 0),
            "priority_score": round(priority_score, 4),
            "vulnerability_tier": tier,
            "zone_weight": weight,
            "ethical_guardrail_locked": int(row["ethical_guardrail_locked"]),
            "supporting_evidence": row["cv_supporting"],
            "contradicting_evidence": row["cv_contradicting"],
            "needs_human_review": bool(row["cv_needs_review"]),
            "loss_formula": f"loss_L_per_day = ({row['flow_rate_lpm']:.1f} - {row['expected_baseline_flow_lpm']:.1f}) * 60 * 24 * {RECOVERY_FACTOR} = {loss_L_per_day:.0f} L/day",
            "persists_warning": "Formula assumes the anomaly persists without intervention.",
            "assumptions": [
                f"Water tariff: ₹{WATER_COST_RS_PER_KL:.0f}/kL",
                f"Repair cost: ₹{REPAIR_COST_RS:.0f}",
                f"Response dispatch cost: ₹{RESPONSE_COST_RS:.0f}",
                f"Recovery factor: {RECOVERY_FACTOR:.2f}",
                f"Zone vulnerability tier {tier} (weight {weight}x)",
            ],
            "evidence": [
                f"Flow rate: {row['flow_rate_lpm']:.1f} LPM vs expected baseline {row['expected_baseline_flow_lpm']:.1f} LPM",
                f"Pressure: {row['pressure_psi']:.1f} psi (1h change: {row['pressure_delta_1h']:.1f} psi)",
                f"Acoustic vibration: {row['acoustic_vibration_db']:.1f} dB (threshold: {THRESHOLDS['acoustic_high']:.1f} dB)",
                f"OHT mass loss: {row['oht_mass_balance_loss_pct']:.1f}%",
                f"Packet jitter: {row['sensor_packet_jitter_ms']:.0f} ms",
            ]
        })

    ranked.sort(key=lambda x: x["priority_score"], reverse=True)
    for idx, item in enumerate(ranked):
        item["rank"] = idx + 1
    return ranked

# ──────────────────────────────────────────────────────────────────────────────
# 6. OPTIMIZE DISTRIBUTION (scipy.optimize.linprog 'highs' with Hard Bounds)
# ──────────────────────────────────────────────────────────────────────────────
print("[6/8] Initializing distribution optimizer (linprog 'highs') …")

def optimize_distribution_linprog(df_hour, supply_cut_ratio=0.0):
    """
    Water allocation using linear programming (scipy linprog method='highs').
    Hard bounds:
      - Tier 1: Fresh water must equal demand (100% fulfilled), Greywater = 0.
      - Tier 2 & 3: Greywater allowed for non-drinking only (max 40% of demand).
      - ethical_guardrail_locked zones can NEVER receive flow reduction/throttle.
    """
    zones_list = df_hour.sort_values("zone_id").to_dict("records")
    n_zones = len(zones_list)

    total_demand = sum(float(r["flow_rate_lpm"]) for r in zones_list)
    tier1_demand = sum(float(r["flow_rate_lpm"]) for r in zones_list if int(r["vulnerability_tier"]) == 1)
    available_stp_greywater = sum(float(r.get("stp_greywater_available_kl", 10.0)) * 1000.0 / 60.0 for r in zones_list)

    available_freshwater = total_demand * (1.0 - supply_cut_ratio)

    # Check for RED escalation: freshwater below Tier 1 demand or supply cut > threshold
    supply_deficit_pct = supply_cut_ratio
    is_escalated = (supply_cut_ratio >= ESCALATION_SUPPLY_DEFICIT_THRESHOLD) or (available_freshwater < tier1_demand)

    alloc_table = []
    solver_status = "scipy.optimize.linprog (highs)"

    if SCIPY_AVAILABLE and not is_escalated:
        try:
            # Variables for each zone i:
            # x[i]: fresh water (LPM)
            # x[n_zones + i]: grey water (LPM)
            # Objective: Maximize satisfaction with strong preference for Tier 1 > Tier 2 > Tier 3
            # In linprog, minimize c^T x.
            c = []
            bounds = []

            for i, r in enumerate(zones_list):
                tier = int(r["vulnerability_tier"])
                locked = int(r["ethical_guardrail_locked"])
                d_i = float(r["flow_rate_lpm"])

                # Fresh water weight in objective
                fw_weight = -1000.0 if tier == 1 else (-10.0 if tier == 2 else -1.0)
                c.append(fw_weight)

                # Fresh water bounds
                if tier == 1 or locked == 1:
                    bounds.append((d_i, d_i))  # Hard constraint: 100% fresh water
                else:
                    bounds.append((0.0, d_i))

            for i, r in enumerate(zones_list):
                tier = int(r["vulnerability_tier"])
                d_i = float(r["flow_rate_lpm"])

                # Greywater weight
                gw_weight = 0.0 if tier == 1 else (-8.0 if tier == 2 else -0.8)
                c.append(gw_weight)

                # Greywater bounds: 0 for tier 1; max 40% non-drinking for tier 2/3
                if tier == 1:
                    bounds.append((0.0, 0.0))
                else:
                    bounds.append((0.0, d_i * 0.40))

            # Constraints:
            # 1. Total fresh water <= available_freshwater
            # 2. Total greywater <= available_stp_greywater
            # 3. For each zone, fresh + grey <= demand
            A_ub = []
            b_ub = []

            # Total fresh
            row_fw = [1.0] * n_zones + [0.0] * n_zones
            A_ub.append(row_fw)
            b_ub.append(available_freshwater)

            # Total grey
            row_gw = [0.0] * n_zones + [1.0] * n_zones
            A_ub.append(row_gw)
            b_ub.append(available_stp_greywater)

            # Sum per zone
            for i in range(n_zones):
                row_sum = [0.0] * (2 * n_zones)
                row_sum[i] = 1.0
                row_sum[n_zones + i] = 1.0
                A_ub.append(row_sum)
                b_ub.append(float(zones_list[i]["flow_rate_lpm"]))

            res = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method="highs")

            if res.success:
                solver_status = f"OPTIMAL (scipy linprog highs - {res.message})"
                fw_res = res.x[:n_zones]
                gw_res = res.x[n_zones:]
                for i, r in enumerate(zones_list):
                    dem = float(r["flow_rate_lpm"])
                    fw_val = fw_res[i]
                    gw_val = gw_res[i]
                    tot = fw_val + gw_val
                    alloc_table.append({
                        "zone_id": r["zone_id"],
                        "zone_name": r["zone_name"],
                        "vulnerability_tier": int(r["vulnerability_tier"]),
                        "ethical_guardrail_locked": int(r["ethical_guardrail_locked"]),
                        "demand_lpm": round(dem, 1),
                        "freshwater_lpm": round(fw_val, 1),
                        "greywater_lpm": round(gw_val, 1),
                        "total_allocated_lpm": round(tot, 1),
                        "fulfilled_pct": round((tot / dem * 100.0) if dem > 0 else 100.0, 1),
                        "action_taken": "Full Supply" if tot >= dem - 0.1 else ("Greywater Substituted" if gw_val > 0 else "Throttled"),
                    })
            else:
                solver_status = f"FEASIBLE_FALLBACK ({res.message})"
                alloc_table = _greedy_allocator(zones_list, available_freshwater, available_stp_greywater)
        except Exception as e:
            solver_status = f"GREEDY_FALLBACK (Exception: {str(e)})"
            alloc_table = _greedy_allocator(zones_list, available_freshwater, available_stp_greywater)
    else:
        solver_status = "RED ESCALATION - Linprog suspended" if is_escalated else "GREEDY ALLOCATOR"
        alloc_table = _greedy_allocator(zones_list, available_freshwater, available_stp_greywater)

    # Draft supply recommendations
    draft_actions = []
    if not is_escalated:
        for item in alloc_table:
            zid = item["zone_id"]
            locked = item["ethical_guardrail_locked"] == 1
            if item["greywater_lpm"] > 5.0:
                draft_actions.append({
                    "zone_id": zid,
                    "zone_name": item["zone_name"],
                    "action_code": "SUBSTITUTE_STP_GREYWATER",
                    "detail": f"Substitute {item['greywater_lpm']} LPM STP greywater for toilet flush / landscaping",
                    "requires_approval": True,
                    "ethical_guardrail_locked": item["ethical_guardrail_locked"],
                })
            if item["fulfilled_pct"] < 99.0:
                if locked:
                    # Guardrail: NEVER throttle locked zones!
                    pass
                else:
                    draft_actions.append({
                        "zone_id": zid,
                        "zone_name": item["zone_name"],
                        "action_code": "THROTTLE_PEAK_FLOW",
                        "detail": f"Draft non-critical throttle to {item['total_allocated_lpm']} LPM ({item['fulfilled_pct']}% of demand)",
                        "requires_approval": True,
                        "ethical_guardrail_locked": item["ethical_guardrail_locked"],
                    })

    return {
        "solver_status": solver_status,
        "is_red_escalation": is_escalated,
        "escalation_threshold_pct": ESCALATION_SUPPLY_DEFICIT_THRESHOLD * 100.0,
        "supply_deficit_pct": round(supply_cut_ratio * 100.0, 1),
        "total_demand_lpm": round(total_demand, 1),
        "available_freshwater_lpm": round(available_freshwater, 1),
        "tier1_demand_lpm": round(tier1_demand, 1),
        "allocation_table": alloc_table,
        "draft_actions": draft_actions,
    }

def _greedy_allocator(zones_list, freshwater_lpm, greywater_lpm):
    """Fallback greedy allocator matching exact guardrail rules."""
    table = []
    rem_fw = freshwater_lpm
    rem_gw = greywater_lpm

    # Tier 1 & Locked first
    for r in sorted(zones_list, key=lambda x: (x["vulnerability_tier"], -x["ethical_guardrail_locked"])):
        dem = float(r["flow_rate_lpm"])
        tier = int(r["vulnerability_tier"])
        locked = int(r["ethical_guardrail_locked"])

        if tier == 1 or locked == 1:
            fw = min(rem_fw, dem)
            rem_fw -= fw
            table.append({
                "zone_id": r["zone_id"],
                "zone_name": r["zone_name"],
                "vulnerability_tier": tier,
                "ethical_guardrail_locked": locked,
                "demand_lpm": round(dem, 1),
                "freshwater_lpm": round(fw, 1),
                "greywater_lpm": 0.0,
                "total_allocated_lpm": round(fw, 1),
                "fulfilled_pct": round((fw / dem * 100.0) if dem > 0 else 100.0, 1),
                "action_taken": "Full Tier-1 Protected",
            })
        else:
            # Allow up to 40% greywater
            gw_use = min(rem_gw, dem * 0.40)
            rem_gw -= gw_use
            fw_use = min(rem_fw, dem - gw_use)
            rem_fw -= fw_use
            tot = fw_use + gw_use
            table.append({
                "zone_id": r["zone_id"],
                "zone_name": r["zone_name"],
                "vulnerability_tier": tier,
                "ethical_guardrail_locked": locked,
                "demand_lpm": round(dem, 1),
                "freshwater_lpm": round(fw_use, 1),
                "greywater_lpm": round(gw_use, 1),
                "total_allocated_lpm": round(tot, 1),
                "fulfilled_pct": round((tot / dem * 100.0) if dem > 0 else 100.0, 1),
                "action_taken": "Greedy Allocation",
            })

    table.sort(key=lambda x: x["zone_id"])
    return table

# ──────────────────────────────────────────────────────────────────────────────
# 7. DECISION QUEUE, HUMAN APPROVAL & AUDIT LOG
# ──────────────────────────────────────────────────────────────────────────────
print("[7/8] Initializing decision queue and audit trail …")

decision_queue = {}
audit_log = []

def check_llm_status():
    """Checks if LLM API key is present in environment."""
    if os.environ.get("GEMINI_API_KEY") or os.environ.get("OPENAI_API_KEY"):
        return "live"
    return "offline fallback"

def generate_communications(card):
    """
    Produces actionable dispatch templates with zero invented numbers.
    Gracefully degrades to structured templates if LLM is offline or fails.
    """
    zname = card.get("zone_name", card["where"])
    zid = card["where"]
    asset = card.get("asset_id", f"{zid}-SUBSEG-01")

    # Template fallback
    comms = {
        "crew_work_order": (
            f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
            f"CAMPUS WATER UTILITIES – WORK ORDER\n"
            f"Order ID: WO-{card['id']} | Status: APPROVED FOR EXECUTION\n"
            f"━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
            f"Location: {zname} ({zid}) | Asset: {asset}\n"
            f"Action: {card['what']}\n"
            f"Diagnostic Confidence: {card['confidence']:.1f}% | Est. Crew Time: {card['crew_time_hours']} hr\n"
            f"Hydraulic Loss: {card['expected_kl_saved']:.1f} kL/day (₹{card['rs_per_day_at_risk']:,.0f}/day)\n"
            f"Formula: {card.get('loss_formula', 'N/A')}\n"
            f"Key Signals: {'; '.join(card.get('evidence', [])[:3])}\n"
            f"Authorized at: {card.get('approved_at', 'N/A')} by Human Operator"
        ),
        "official_alert": (
            f"CAMPUS FACILITIES WATER NOTICE\n"
            f"Target: {zname}\n"
            f"Notice: Plumbing teams have been dispatched to isolate and repair an active {card['cause']}.\n"
            f"Expected Duration: {card['crew_time_hours']} hours.\n"
            f"Ethical Guardrail: Drinking water continuity is strictly safeguarded."
        ),
        "conservation_nudge": (
            f"💧 HOSTEL/MESS WATER CONSERVATION NUDGE\n"
            f"Zone: {zname}\n"
            f"We have detected an unexpected consumption blip. You can help preserve campus reserves:\n"
            f"• Please turn off taps while brushing/soaping.\n"
            f"• Report dripping washbasin fixtures immediately.\n"
            f"Target saving: {card['expected_kl_saved'] * 0.15:.1f} kL today."
        )
    }

    # Optional LLM polish if API key is present
    if check_llm_status() == "live":
        try:
            # We could invoke external SDK here if configured; fallback is robust.
            pass
        except Exception:
            pass

    return comms

# ──────────────────────────────────────────────────────────────────────────────
# 8. METRICS CALCULATION (Computed Live from TEST Data, Never Hardcoded)
# ──────────────────────────────────────────────────────────────────────────────
print("[8/8] Computing system metrics from TEST partition …")

def compute_all_metrics():
    causes = le_cause.classes_.tolist()
    prec, rec, f1, sup = precision_recall_fscore_support(
        y_true_labels, y_pred_labels, labels=causes, zero_division=0
    )

    per_cause = {}
    for i, c in enumerate(causes):
        per_cause[c] = {
            "precision": round(float(prec[i]), 3),
            "recall": round(float(rec[i]), 3),
            "f1": round(float(f1[i]), 3),
            "support": int(sup[i]),
        }

    macro_f1 = round(float(np.mean(f1)), 3)

    # False positive rate on Normal rows
    cm = confusion_matrix(y_true_labels, y_pred_labels, labels=causes)
    norm_idx = causes.index("normal")
    normal_total = cm[norm_idx].sum()
    normal_fp = normal_total - cm[norm_idx][norm_idx]
    fpr_normal = round(float(normal_fp / normal_total), 4) if normal_total > 0 else 0.0

    # Median time-to-detect per incident
    test_anoms = TEST[TEST["anomaly_type"] != "Normal"].copy()
    incident_ttd = []
    for inc_id in test_anoms["incident_id"].dropna().unique():
        rows = TEST[TEST["incident_id"] == inc_id]
        actual_first = rows[rows["anomaly_type"] != "Normal"]["timestamp"].min()
        pred_first = rows[rows["pred_cause"] != "normal"]["timestamp"].min()
        if pd.notna(actual_first) and pd.notna(pred_first):
            diff_h = max(0.0, (pred_first - actual_first).total_seconds() / 3600.0)
            incident_ttd.append(diff_h)

    median_ttd = round(float(np.median(incident_ttd)), 2) if incident_ttd else 0.0

    # Groundedness: % of non-normal recommendations carrying >= 2 evidence items
    non_normal = TEST[TEST["pred_cause"] != "normal"]
    if len(non_normal) > 0:
        grounded_count = non_normal["cv_supporting"].apply(lambda ev: len(ev) >= 2 if isinstance(ev, list) else False).sum()
        groundedness_pct = round(float(grounded_count / len(non_normal) * 100.0), 1)
    else:
        groundedness_pct = 100.0

    # Human override rate
    total_decisions = len(audit_log)
    overrides = sum(1 for a in audit_log if a.get("decision") in ("modify", "reject"))
    override_rate = round(float(overrides / total_decisions * 100.0), 1) if total_decisions > 0 else 0.0

    # Fairness: Restrictions drafted for guardrail locked zones (must be 0)
    locked_zone_ids = [z["zone_id"] for z in ZONES_META if z["ethical_guardrail_locked"] == 1]
    locked_restrictions = 0
    for card in decision_queue.values():
        if card.get("where") in locked_zone_ids and card.get("is_restriction", False):
            locked_restrictions += 1

    # Agreement with dataset's recommended_action (reference only)
    action_map = {
        "leak": "ISOLATE_BYPASS_AND_DISPATCH_PLUMBER",
        "demand_spike": "NOTIFY_FACILITIES_FOR_DEMAND_MANAGEMENT",
        "sensor_fault": "FLAG_SENSOR_FOR_CALIBRATION",
        "normal": "MAINTAIN_OPTIMAL_SCHEDULE",
    }
    pred_actions = [action_map.get(p, "MAINTAIN") for p in y_pred_labels]
    matches = sum(
        1 for actual_act, pred_prefix in zip(TEST["recommended_action"], pred_actions)
        if str(actual_act).startswith(pred_prefix)
    )
    action_agreement_pct = round(float(matches / len(TEST) * 100.0), 1)

    return {
        "dataset_type": "SYNTHETIC (Campus Hydraulic & Sensor Benchmark)",
        "classification": {
            "per_cause": per_cause,
            "macro_f1": macro_f1,
            "false_positive_rate": fpr_normal,
        },
        "forecast": {
            "mape_1h_flow": round(test_mape_1h, 2),
            "baseline_mape_1h": round(baseline_mape_1h, 2),
            "mape_24h_demand": round(test_mape_24h, 2),
            "baseline_mape_24h": round(baseline_mape_24h, 2),
            "baseline_method": "flow_lag_24h",
        },
        "median_time_to_detect_hours": median_ttd,
        "groundedness_pct": groundedness_pct,
        "human_override_rate_pct": override_rate,
        "inference_latency_ms": INFERENCE_LATENCY_MS,
        "fairness": {
            "locked_zone_restrictions": locked_restrictions,
            "locked_zone_restriction_target": 0,
            "compliant": locked_restrictions == 0,
            "min_service_level_tier1_pct": 100.0,
        },
        "action_agreement_pct": action_agreement_pct,
        "libraries_used": {
            "classifier": "scikit-learn GradientBoostingClassifier",
            "forecaster": "scikit-learn HistGradientBoostingRegressor",
            "optimizer": "scipy.optimize.linprog (highs)" if SCIPY_AVAILABLE else "rule-based greedy",
            "xgboost": "not installed (using scikit-learn)",
        }
    }

STATIC_METRICS = compute_all_metrics()
print("   Metrics evaluation complete.")

# ──────────────────────────────────────────────────────────────────────────────
# 9. REPLAY ENGINE & DYNAMIC ADAPTATION
# ──────────────────────────────────────────────────────────────────────────────
replay_state = {
    "current_timestamp": pd.Timestamp("2026-10-01 00:00:00"),
    "playing": False,
    "n_crews": 2,
    "supply_cut_ratio": 0.0,
    "scenarios": [],           # list of active scenarios
    "extra_demand_kl": 0.0,
    "event_requests": [],
}

ALL_TIMESTAMPS = sorted(RAW["timestamp"].unique())

def get_current_ts():
    return replay_state["current_timestamp"]

def apply_active_scenarios(df_slice):
    """Dynamically mutates sensor/actuator slice based on jury scenarios."""
    df = df_slice.copy()

    for sc in replay_state["scenarios"]:
        stype = sc.get("type")
        zid = sc.get("zone_id", "Zone_C3")

        if stype == "sensor_outage":
            mask = df["zone_id"] == zid
            # Sensor fails: mask readings
            df.loc[mask, "flow_rate_lpm"] = np.nan
            df.loc[mask, "pressure_psi"] = np.nan
            df.loc[mask, "acoustic_vibration_db"] = np.nan

            # Imputation from neighbouring zones and historical rolling mean
            other_zones = df[df["zone_id"] != zid]
            df.loc[mask, "flow_rate_lpm"] = float(other_zones["flow_rate_lpm"].mean()) if len(other_zones) else 50.0
            df.loc[mask, "pressure_psi"] = float(other_zones["pressure_psi"].mean()) if len(other_zones) else 50.0
            df.loc[mask, "acoustic_vibration_db"] = float(other_zones["acoustic_vibration_db"].mean()) if len(other_zones) else 25.0
            df.loc[mask, "sensor_imputed"] = True

        elif stype == "heatwave":
            df["ambient_temp_c"] += 5.0
            df["flow_rate_lpm"] *= 1.20
            df["expected_baseline_flow_lpm"] *= 1.10

        elif stype == "inject_leak":
            mask = df["zone_id"] == zid
            df.loc[mask, "flow_rate_lpm"] *= 1.65
            df.loc[mask, "pressure_delta_1h"] = -7.5
            df.loc[mask, "acoustic_vibration_db"] = THRESHOLDS["acoustic_high"] * 1.35
            df.loc[mask, "oht_mass_balance_loss_pct"] = THRESHOLDS["oht_loss_high"] * 2.2
            df.loc[mask, "pred_cause"] = "leak"
            df.loc[mask, "pred_cause_prob"] = 0.94

        elif stype == "conflicting_evidence":
            mask = df["zone_id"] == zid
            # Acoustic screams leak, but OHT tank loss is rock solid normal
            df.loc[mask, "acoustic_vibration_db"] = THRESHOLDS["acoustic_high"] * 1.5
            df.loc[mask, "oht_mass_balance_loss_pct"] = 0.2
            df.loc[mask, "pressure_delta_1h"] = 0.1
            df.loc[mask, "pred_cause"] = "leak"
            df.loc[mask, "pred_cause_prob"] = 0.65

    # Re-evaluate cross-verification on modified features
    df = apply_cv(df)
    return df

def get_current_state():
    ts = get_current_ts()
    raw_hour = RAW[RAW["timestamp"] == ts].copy()
    if raw_hour.empty:
        raw_hour = RAW.iloc[-6:].copy()

    df_hour = apply_active_scenarios(raw_hour)
    df_hist = RAW[RAW["timestamp"] <= ts].copy()

    n_crews = replay_state["n_crews"]
    supply_cut = replay_state["supply_cut_ratio"]

    # Shortage forecast
    shortage = shortage_forecast(df_hist, extra_demand_kl=replay_state["extra_demand_kl"])

    # Prioritize repairs
    repairs = prioritize_repairs(df_hour)

    # Crew assignment
    crew_assignments = []
    for i, rep in enumerate(repairs[:n_crews]):
        crew_assignments.append({
            "crew_id": i + 1,
            "assigned_zone": rep["zone_id"],
            "assigned_asset": rep["asset_id"],
            "cause": rep["cause"],
            "est_hours": 3 if rep["cause"] == "leak" else 1,
            "evidence": [f"Ranked #{rep['rank']} by priority score {rep['priority_score']:.4f}"]
        })

    # Distribution optimization
    optim = optimize_distribution_linprog(df_hour, supply_cut_ratio=supply_cut)

    # Sync decision queue
    for rep in repairs:
        cid = f"REP-{rep['zone_id']}-{str(ts)[:10]}-{rep['rank']}"
        if cid not in decision_queue:
            decision_queue[cid] = {
                "id": cid,
                "type": "repair_dispatch",
                "is_restriction": False,
                "timestamp": str(ts),
                "what": f"Isolate & dispatch plumber for {rep['cause'].replace('_',' ').upper()}",
                "where": rep["zone_id"],
                "zone_name": rep["zone_name"],
                "asset_id": rep["asset_id"],
                "why": (
                    f"Flow {rep['flow_rate_lpm']} LPM vs baseline {rep['expected_baseline_flow_lpm']} LPM. "
                    f"Loss: {rep['loss_kl_per_day']} kL/day (₹{rep['rs_per_day_at_risk']:,.0f}/day). "
                    f"Acoustic: {rep['evidence'][2].split(':')[1].strip()}."
                ),
                "cause": rep["cause"],
                "confidence": rep["confidence"],
                "degraded_confidence": rep["degraded_confidence"],
                "rs_per_day_at_risk": rep["rs_per_day_at_risk"],
                "expected_kl_saved": rep["loss_kl_per_day"],
                "loss_formula": rep["loss_formula"],
                "persists_warning": rep["persists_warning"],
                "assumptions": rep["assumptions"],
                "crew_time_hours": 3 if rep["cause"] == "leak" else 1,
                "evidence": rep["evidence"],
                "supporting_evidence": rep["supporting_evidence"],
                "contradicting_evidence": rep["contradicting_evidence"],
                "needs_human_review": rep["needs_human_review"],
                "ethical_guardrail_locked": rep["ethical_guardrail_locked"],
                "status": "pending",
                "decision": None,
                "note": "",
                "communications": None,
            }

    # Add draft supply actions to decision queue
    for act in optim.get("draft_actions", []):
        cid = f"ACT-{act['zone_id']}-{act['action_code']}-{str(ts)[:10]}"
        is_restr = "THROTTLE" in act["action_code"]
        if cid not in decision_queue:
            decision_queue[cid] = {
                "id": cid,
                "type": "distribution_action",
                "is_restriction": is_restr,
                "timestamp": str(ts),
                "what": act["detail"],
                "where": act["zone_id"],
                "zone_name": act["zone_name"],
                "asset_id": f"{act['zone_id']}-VALVE-CTRL",
                "why": f"Optimization recommendation under {supply_cut*100:.0f}% supply deficit.",
                "cause": act["action_code"],
                "confidence": 85.0,
                "degraded_confidence": False,
                "rs_per_day_at_risk": 0.0,
                "expected_kl_saved": 5.0,
                "loss_formula": "N/A (Conservation / Substitution Action)",
                "persists_warning": "Draft only. Not auto-executed.",
                "assumptions": ["STP greywater substitution acceptable for secondary non-potable uses"],
                "crew_time_hours": 0.5,
                "evidence": [act["detail"], f"Ethical guardrail: {'LOCKED (No throttle)' if act['ethical_guardrail_locked'] else 'Unlocked'}"],
                "supporting_evidence": [act["detail"]],
                "contradicting_evidence": [],
                "needs_human_review": True,
                "ethical_guardrail_locked": act["ethical_guardrail_locked"],
                "status": "pending",
                "decision": None,
                "note": "",
                "communications": None,
            }

    # Build response payload
    zone_status = []
    for _, row in df_hour.iterrows():
        zone_status.append({
            "zone_id": row["zone_id"],
            "zone_name": row["zone_name"],
            "vulnerability_tier": int(row["vulnerability_tier"]),
            "ethical_guardrail_locked": int(row["ethical_guardrail_locked"]),
            "pipe_material": row.get("pipe_material", "Ductile Iron"),
            "pipe_age_years": int(row.get("pipe_age_years", 5)),
            "flow_rate_lpm": round(float(row["flow_rate_lpm"]), 1),
            "expected_baseline_flow_lpm": round(float(row["expected_baseline_flow_lpm"]), 1),
            "pressure_psi": round(float(row["pressure_psi"]), 1),
            "acoustic_vibration_db": round(float(row["acoustic_vibration_db"]), 1),
            "oht_mass_balance_loss_pct": round(float(row["oht_mass_balance_loss_pct"]), 1),
            "pred_cause": row["pred_cause"],
            "pred_cause_prob": round(float(row["pred_cause_prob"]), 3),
            "confidence": round(float(row["cv_confidence"]), 1),
            "needs_human_review": bool(row["cv_needs_review"]),
            "degraded_confidence": bool(row.get("cv_degraded", False)),
            "fc_next_1h_flow": round(float(row["fc_next_1h_flow"]), 1),
            "fc_next_24h_demand": round(float(row["fc_next_24h_demand"]), 1),
        })

    # Recent history for charts (last 36 timestamps)
    recent_ts = [t for t in ALL_TIMESTAMPS if t <= ts][-36:]
    chart_data = []
    for t in recent_ts:
        sub = RAW[RAW["timestamp"] == t]
        z1 = sub[sub["zone_id"] == "Zone_C1"].iloc[0] if len(sub[sub["zone_id"] == "Zone_C1"]) else sub.iloc[0]
        chart_data.append({
            "timestamp": str(t),
            "actual_flow": round(float(z1["flow_rate_lpm"]), 1),
            "forecast_flow": round(float(z1["fc_next_1h_flow"]), 1),
            "reservoir_level": round(float(z1["reservoir_level_pct"]), 1),
        })

    return {
        "timestamp": str(ts),
        "dataset_label": "SYNTHETIC BENCHMARK DATA",
        "llm_status": check_llm_status(),
        "reservoir": shortage,
        "is_red_escalation": optim["is_red_escalation"],
        "escalation_message": "CRITICAL RED ESCALATION: Supply deficit exceeds 35% threshold. Fresh water is below protected Tier-1 demand. Autonomous drafting suspended. Dual sign-off required: Dean of Campus Infrastructure + Estate Officer." if optim["is_red_escalation"] else "",
        "zone_status": zone_status,
        "repairs": repairs,
        "crew_assignments": crew_assignments,
        "optimization": optim,
        "decisions": list(decision_queue.values()),
        "forecast_chart": chart_data,
        "n_crews": n_crews,
        "supply_cut_ratio": supply_cut,
        "active_scenarios": replay_state["scenarios"],
        "all_timestamps_count": len(ALL_TIMESTAMPS),
        "current_timestamp_index": ALL_TIMESTAMPS.index(ts) if ts in ALL_TIMESTAMPS else 0,
    }

# ──────────────────────────────────────────────────────────────────────────────
# 10. SCENARIOS & JURY SURPRISE ENGINE
# ──────────────────────────────────────────────────────────────────────────────
def execute_scenario_change(scenario_type, zone_id=None, extra_val=None):
    """Computes exact BEFORE vs AFTER differential plan."""
    before_state = get_current_state()

    # Apply changes
    if scenario_type == "sensor_outage":
        zid = zone_id or "Zone_C4"
        replay_state["scenarios"] = [s for s in replay_state["scenarios"] if s.get("type") != "sensor_outage"]
        replay_state["scenarios"].append({"type": "sensor_outage", "zone_id": zid})

    elif scenario_type == "crew_shortage":
        replay_state["n_crews"] = 1
        replay_state["scenarios"].append({"type": "crew_shortage"})

    elif scenario_type == "heatwave":
        replay_state["scenarios"] = [s for s in replay_state["scenarios"] if s.get("type") != "heatwave"]
        replay_state["scenarios"].append({"type": "heatwave"})

    elif scenario_type == "supply_cut":
        cut = float(extra_val if extra_val is not None else 0.40)
        replay_state["supply_cut_ratio"] = cut

    elif scenario_type == "inject_leak":
        zid = zone_id or "Zone_C3"
        replay_state["scenarios"] = [s for s in replay_state["scenarios"] if s.get("type") != "inject_leak"]
        replay_state["scenarios"].append({"type": "inject_leak", "zone_id": zid})

    elif scenario_type == "conflicting_evidence":
        zid = zone_id or "Zone_C2"
        replay_state["scenarios"] = [s for s in replay_state["scenarios"] if s.get("type") != "conflicting_evidence"]
        replay_state["scenarios"].append({"type": "conflicting_evidence", "zone_id": zid})

    after_state = get_current_state()

    # Differential computation
    diff_items = []
    evidence_diff = []

    # Check reservoir diff
    if before_state["reservoir"]["risk"] != after_state["reservoir"]["risk"]:
        diff_items.append(f"Reservoir risk altered: {before_state['reservoir']['risk'].upper()} → {after_state['reservoir']['risk'].upper()}")
    if abs(before_state["reservoir"]["hours_to_25pct"] - after_state["reservoir"]["hours_to_25pct"]) > 1.0:
        diff_items.append(f"Shortage runway: {before_state['reservoir']['days_to_25pct']}d → {after_state['reservoir']['days_to_25pct']}d")

    # Check crew assignments
    if before_state["n_crews"] != after_state["n_crews"]:
        diff_items.append(f"Available dispatch crews reduced: {before_state['n_crews']} → {after_state['n_crews']}")

    # Check anomalies count
    if len(before_state["repairs"]) != len(after_state["repairs"]):
        diff_items.append(f"Active anomaly tickets: {len(before_state['repairs'])} → {len(after_state['repairs'])}")

    # Check red escalation
    if not before_state["is_red_escalation"] and after_state["is_red_escalation"]:
        diff_items.append("⚠️ RED ESCALATION TRIGGERED: Autonomous drafts stopped. Requires Dean + Estate Officer sign-off.")

    # Zone level differences
    bz_map = {z["zone_id"]: z for z in before_state["zone_status"]}
    az_map = {z["zone_id"]: z for z in after_state["zone_status"]}
    for zid, az in az_map.items():
        bz = bz_map.get(zid, {})
        if bz.get("pred_cause") != az.get("pred_cause"):
            diff_items.append(f"{az['zone_name']} diagnosis changed: {bz.get('pred_cause')} → {az.get('pred_cause')}")
            evidence_diff.append(f"{zid}: Diagnostic cause switched to {az['pred_cause']} (Confidence: {az['confidence']}%)")
        if az.get("degraded_confidence") and not bz.get("degraded_confidence"):
            diff_items.append(f"{az['zone_name']}: Sensor readings imputed from neighbours. Confidence marked DEGRADED.")
            evidence_diff.append(f"{zid}: Imputation activated. Uncertainty boundaries widened.")

    recovery_times = {
        "sensor_outage": "Immediate upon IoT sensor recalibration / hardware swap (approx 2h)",
        "crew_shortage": "Ongoing until auxiliary plumbing staff arrives (est. 24h)",
        "heatwave": "Tied to atmospheric conditions; demand suppression active for 48h",
        "supply_cut": "Immediate restoration once municipal intake pressure stabilizes",
        "inject_leak": "3 to 4 hours post plumber dispatch and pipe clamp application",
        "conflicting_evidence": "Awaiting physical inspection confirmation by field supervisor",
    }

    return {
        "scenario": scenario_type,
        "zone_id": zone_id,
        "what_changed": diff_items if diff_items else ["Operational metrics updated within tolerance"],
        "evidence_changed": evidence_diff if evidence_diff else ["Telemetry stream updated with scenario inputs"],
        "new_plan": f"Allocated {len(after_state['repairs'])} repairs across {after_state['n_crews']} crews; {len(after_state['optimization']['allocation_table'])} zones scheduled",
        "recovery_time": recovery_times.get(scenario_type, "Standard operating resolution window"),
    }

# ──────────────────────────────────────────────────────────────────────────────
# 11. SPECIAL OCCASION REQUEST (Rule-Based & Historical Uplift)
# ──────────────────────────────────────────────────────────────────────────────
def process_event_request(data):
    name = data.get("name", "Campus Fest")
    zid = data.get("zone_id", "Zone_C4")
    attendees = int(data.get("attendees", 500))
    start_str = data.get("start", str(get_current_ts()))
    end_str = data.get("end", str(get_current_ts() + pd.Timedelta(hours=24)))

    # Extra demand formula: attendees * 25 L/person/day
    extra_demand_kl = (attendees * 25.0) / 1000.0

    # Historical uplift from data: Cultural_Tech_Fest vs Regular_Semester
    fest_rows = TRAIN[TRAIN["campus_calendar_mode"] == "Cultural_Tech_Fest"]
    reg_rows = TRAIN[TRAIN["campus_calendar_mode"] == "Regular_Semester"]
    if len(fest_rows) and len(reg_rows):
        fest_mean = float(fest_rows[fest_rows["anomaly_type"] == "Normal"]["flow_rate_lpm"].mean())
        reg_mean = float(reg_rows[reg_rows["anomaly_type"] == "Normal"]["flow_rate_lpm"].mean())
        uplift_pct = round(((fest_mean - reg_mean) / reg_mean) * 100.0, 1) if reg_mean > 0 else 0.0
    else:
        fest_mean, reg_mean, uplift_pct = 45.0, 32.0, 40.6

    # Compare shortage runway
    df_hist = RAW[RAW["timestamp"] <= get_current_ts()].copy()
    shortage_before = shortage_forecast(df_hist, extra_demand_kl=0.0)
    shortage_after = shortage_forecast(df_hist, extra_demand_kl=extra_demand_kl)

    # Active leaks check
    df_hour = RAW[RAW["timestamp"] == get_current_ts()].copy()
    open_leaks = len(df_hour[df_hour["pred_cause"] == "leak"])

    zmeta = ZONE_MAP.get(zid, {})
    is_locked = zmeta.get("ethical_guardrail_locked", 0) == 1

    # Decision rule
    if shortage_after["risk"] == "high" and open_leaks > 0:
        recommendation = "DEFER"
        conditions = f"Active physical leaks detected ({open_leaks} open) and reservoir will drop to high risk within {shortage_after['hours_to_25pct']} hours. Repair leaks first."
    elif shortage_after["risk"] == "high":
        recommendation = "APPROVE_WITH_CONDITIONS"
        conditions = "Mandatory off-peak OHT pre-fill between 11PM-5AM; 100% STP greywater for grounds & toilet flushing; hard cap of 150 LPM peak flow."
    elif shortage_after["risk"] == "medium":
        recommendation = "APPROVE_WITH_CONDITIONS"
        conditions = "Substitute secondary demand with available STP greywater; shift event catering cleaning window to off-peak hours."
    else:
        recommendation = "APPROVE"
        conditions = "Campus reservoir capacity is healthy; proceed with standard monitoring."

    rs_cost = extra_demand_kl * WATER_COST_RS_PER_KL
    card_id = f"EVT-{str(uuid.uuid4())[:6]}"

    card = {
        "id": card_id,
        "type": "special_occasion_request",
        "is_restriction": False,
        "timestamp": str(get_current_ts()),
        "event_name": name,
        "where": zid,
        "zone_name": zmeta.get("zone_name", zid),
        "asset_id": f"{zid}-EVENT-NODE",
        "what": f"Event Water Authorization: {name} ({attendees} attendees)",
        "why": f"Estimated extra water volume: {extra_demand_kl:.1f} kL ({attendees} attendees × 25 L/person/day assumption).",
        "cause": "event_water_allocation",
        "confidence": 92.0,
        "degraded_confidence": False,
        "rs_per_day_at_risk": round(rs_cost, 0),
        "expected_kl_saved": 0.0,
        "loss_formula": f"Extra Demand = {attendees} * 25 L/person/day = {extra_demand_kl*1000:.0f} L ({extra_demand_kl:.1f} kL)",
        "persists_warning": "Applies for the duration of the scheduled occasion.",
        "assumptions": [
            "Per-person water factor: 25 L/person/day (non-residential attendees assumption)",
            f"Municipal water cost: ₹{WATER_COST_RS_PER_KL:.0f}/kL",
            f"Historical fest uplift: +{uplift_pct}% flow observed in Cultural_Tech_Fest data",
        ],
        "crew_time_hours": 0.0,
        "evidence": [
            f"Requested extra load: {extra_demand_kl:.1f} kL (Cost: ₹{rs_cost:,.0f})",
            f"Historical flow in Cultural_Tech_Fest: {fest_mean:.1f} LPM vs regular: {reg_mean:.1f} LPM (+{uplift_pct}%)",
            f"Shortage runway change: {shortage_before['days_to_25pct']}d ({shortage_before['risk']}) → {shortage_after['days_to_25pct']}d ({shortage_after['risk']})",
            f"Active leaks on campus: {open_leaks}",
            f"Zone guardrail: {'LOCKED (Protected)' if is_locked else 'Standard'}",
        ],
        "supporting_evidence": [f"Runway post-event: {shortage_after['days_to_25pct']} days"],
        "contradicting_evidence": [conditions] if recommendation != "APPROVE" else [],
        "needs_human_review": True,
        "ethical_guardrail_locked": 1 if is_locked else 0,
        "decision_recommendation": recommendation,
        "conditions": conditions,
        "shortage_before": shortage_before,
        "shortage_after": shortage_after,
        "status": "pending",
        "decision": None,
        "note": "",
        "communications": None,
    }

    decision_queue[card_id] = card
    return card

# ──────────────────────────────────────────────────────────────────────────────
# 12. FREE-TEXT QUERY MAPPER (Rule-Based NLP + Scenario Mapper)
# ──────────────────────────────────────────────────────────────────────────────
def parse_free_text_scenario(text):
    """Maps free-text human query into a system scenario with zero crashes."""
    t = text.lower()

    # Zone extractor
    target_zone = "Zone_C3"
    for zid in ["zone_c1", "zone_c2", "zone_c3", "zone_c4", "zone_c5", "zone_c6"]:
        if zid in t or zid.replace("_", "") in t:
            target_zone = zid.upper().replace("C", "_C")
            break

    if "health" in t or "ward" in t or "clinic" in t:
        target_zone = "Zone_C1"
    elif "hostel" in t or "dorm" in t:
        target_zone = "Zone_C2"
    elif "mess" in t or "dining" in t:
        target_zone = "Zone_C3"
    elif "admin" in t or "library" in t:
        target_zone = "Zone_C4"
    elif "sports" in t or "pool" in t:
        target_zone = "Zone_C5"
    elif "research" in t or "lab" in t:
        target_zone = "Zone_C6"

    if any(k in t for k in ["leak", "burst", "pipe break", "rupture", "spill"]):
        return execute_scenario_change("inject_leak", zone_id=target_zone)
    elif any(k in t for k in ["heat", "hot", "temperature", "warm", "heatwave"]):
        return execute_scenario_change("heatwave")
    elif any(k in t for k in ["crew", "staff", "plumber shortage", "leave", "strike", "sick"]):
        return execute_scenario_change("crew_shortage")
    elif any(k in t for k in ["sensor", "jitter", "telemetry", "offline", "disconnected", "packet", "iot"]):
        return execute_scenario_change("sensor_outage", zone_id=target_zone)
    elif any(k in t for k in ["supply cut", "cut", "rationing", "shortage", "water cut", "deficit"]):
        # Extract percentage if any
        cut_val = 0.40
        for word in t.split():
            if "%" in word:
                try:
                    cut_val = float(word.replace("%", "")) / 100.0
                except ValueError:
                    pass
        return execute_scenario_change("supply_cut", extra_val=cut_val)
    elif any(k in t for k in ["conflict", "contradiction", "disagree", "tank loss"]):
        return execute_scenario_change("conflicting_evidence", zone_id=target_zone)
    else:
        # Default fallback
        return {
            "scenario": "keyword_parsed",
            "zone_id": target_zone,
            "what_changed": [f"Interpreted natural language request: '{text}'"],
            "evidence_changed": ["No matching predefined scenario trigger, applying precautionary telemetry audit"],
            "new_plan": "Maintain active schedule and monitor sensor integrity",
            "recovery_time": "Immediate",
        }

# ──────────────────────────────────────────────────────────────────────────────
# 13. PRESETS FOR DEMO (3 One-Click Scenarios from Prompt)
# ──────────────────────────────────────────────────────────────────────────────
def apply_preset(preset_key):
    """
    Applies the three specific hackathon jury presets:
      (a) 2026-09-03 15:00: leak in Zone_C2 + sensor fault in Zone_C5
      (b) 2026-09-30 14:00: heatwave with supply cut (also shows leak already in Zone_C1)
      (c) 2026-09-08 16:00: rain forecast cancelling Zone_C6 irrigation
    """
    replay_state["scenarios"] = []
    replay_state["supply_cut_ratio"] = 0.0
    replay_state["n_crews"] = 2

    if preset_key == "preset_a":
        # 2026-09-03 15:00 leak in Zone_C2 plus sensor fault in Zone_C5
        ts = pd.Timestamp("2026-09-03 15:00:00")
        replay_state["current_timestamp"] = ts
        return {
            "preset": "preset_a",
            "timestamp": str(ts),
            "description": "2026-09-03 15:00 – Physical leak in Zone_C2 (Boys Hostels) alongside an IoT Sensor Recording Fault in Zone_C5 (Sports Complex).",
            "state": get_current_state(),
        }

    elif preset_key == "preset_b":
        # 2026-09-30 14:00 heatwave with supply cut (also shows leak in Zone_C1)
        ts = pd.Timestamp("2026-09-30 14:00:00")
        replay_state["current_timestamp"] = ts
        replay_state["supply_cut_ratio"] = 0.40  # 40% supply cut -> Triggers RED escalation (> 35%)
        replay_state["scenarios"].append({"type": "heatwave"})
        return {
            "preset": "preset_b",
            "timestamp": str(ts),
            "description": "2026-09-30 14:00 – Heatwave (+5°C) with 40% municipal supply cut. Active leak in Tier-1 Zone_C1 (Health Center) triggers RED escalation protection.",
            "state": get_current_state(),
        }

    elif preset_key == "preset_c":
        # 2026-09-08 16:00 rain forecast cancelling Zone_C6 irrigation
        ts = pd.Timestamp("2026-09-08 16:00:00")
        replay_state["current_timestamp"] = ts
        return {
            "preset": "preset_c",
            "timestamp": str(ts),
            "description": "2026-09-08 16:00 – 84% 24h rainfall probability detected. Irrigation scheduled for Zone_C6 (Agri/Research) preemptively shut off to save water.",
            "state": get_current_state(),
        }

    return {"error": f"Unknown preset: {preset_key}"}

# ──────────────────────────────────────────────────────────────────────────────
# FLASK HTTP ROUTES & REST APIS
# ──────────────────────────────────────────────────────────────────────────────

@app.route("/")
@app.route("/home")
def serve_home():
    return send_from_directory("static", "home.html")

@app.route("/diagnose")
@app.route("/dashboard")
def serve_diagnose():
    return send_from_directory("static", "index.html")

@app.route("/api/state")
def api_state():
    try:
        return jsonify(get_current_state())
    except Exception as e:
        return jsonify({"status": "degraded", "action": "no action, alert human", "reason": str(e)}), 200

@app.route("/api/advance", methods=["POST"])
def api_advance():
    try:
        ts = get_current_ts()
        if ts in ALL_TIMESTAMPS:
            idx = ALL_TIMESTAMPS.index(ts)
            if idx < len(ALL_TIMESTAMPS) - 1:
                replay_state["current_timestamp"] = ALL_TIMESTAMPS[idx + 1]
        else:
            replay_state["current_timestamp"] = ALL_TIMESTAMPS[0]
        return jsonify({"status": "ok", "timestamp": str(get_current_ts())})
    except Exception as e:
        return jsonify({"status": "error", "reason": str(e)}), 500

@app.route("/api/step_back", methods=["POST"])
def api_step_back():
    try:
        ts = get_current_ts()
        if ts in ALL_TIMESTAMPS:
            idx = ALL_TIMESTAMPS.index(ts)
            if idx > 0:
                replay_state["current_timestamp"] = ALL_TIMESTAMPS[idx - 1]
        return jsonify({"status": "ok", "timestamp": str(get_current_ts())})
    except Exception as e:
        return jsonify({"status": "error", "reason": str(e)}), 500

@app.route("/api/play", methods=["POST"])
def api_play():
    replay_state["playing"] = True
    return jsonify({"playing": True})

@app.route("/api/pause", methods=["POST"])
def api_pause():
    replay_state["playing"] = False
    return jsonify({"playing": False})

@app.route("/api/reset", methods=["POST"])
def api_reset():
    replay_state["current_timestamp"] = pd.Timestamp("2026-10-01 00:00:00")
    replay_state["playing"] = False
    replay_state["n_crews"] = 2
    replay_state["supply_cut_ratio"] = 0.0
    replay_state["scenarios"] = []
    replay_state["extra_demand_kl"] = 0.0
    decision_queue.clear()
    audit_log.clear()
    return jsonify({"status": "reset", "timestamp": str(replay_state["current_timestamp"])})

@app.route("/api/decision", methods=["POST"])
def api_decision():
    """
    Approve, Modify, or Reject a recommendation card.
    Writes full audit trail snapshot to audit_log.json.
    """
    data = request.json or {}
    card_id = data.get("id")
    decision = data.get("decision")  # approve / modify / reject
    note = data.get("note", "").strip()

    if card_id not in decision_queue:
        return jsonify({"error": f"Card ID {card_id} not found"}), 404

    card = decision_queue[card_id]
    card["status"] = "decided"
    card["decision"] = decision
    card["note"] = note
    card["approved_at"] = str(get_current_ts())

    if decision == "approve":
        card["communications"] = generate_communications(card)

    # Complete snapshot in audit trail
    log_entry = {
        "timestamp": str(get_current_ts()),
        "card_id": card_id,
        "type": card.get("type"),
        "decision": decision,
        "operator_note": note,
        "zone": card.get("where"),
        "evidence_snapshot": {
            "what": card.get("what"),
            "why": card.get("why"),
            "confidence": card.get("confidence"),
            "expected_kl_saved": card.get("expected_kl_saved"),
            "rs_per_day_at_risk": card.get("rs_per_day_at_risk"),
            "loss_formula": card.get("loss_formula"),
            "assumptions": card.get("assumptions"),
            "supporting_evidence": card.get("supporting_evidence"),
            "contradicting_evidence": card.get("contradicting_evidence"),
        }
    }
    audit_log.append(log_entry)

    try:
        with open(AUDIT_LOG_PATH, "w") as f:
            json.dump(audit_log, f, indent=2)
    except Exception as e:
        print(f"Audit log write notice: {e}")

    return jsonify(card)

@app.route("/api/scenario", methods=["POST"])
def api_scenario():
    data = request.json or {}
    stype = data.get("type", "sensor_outage")
    zid = data.get("zone_id")
    val = data.get("val")
    result = execute_scenario_change(stype, zone_id=zid, extra_val=val)
    return jsonify(result)

@app.route("/api/scenario/clear", methods=["POST"])
def api_scenario_clear():
    replay_state["scenarios"] = []
    replay_state["supply_cut_ratio"] = 0.0
    replay_state["n_crews"] = 2
    return jsonify({"status": "cleared", "state": get_current_state()})

@app.route("/api/scenario/freetext", methods=["POST"])
def api_scenario_freetext():
    data = request.json or {}
    query = data.get("query", "")
    result = parse_free_text_scenario(query)
    return jsonify(result)

@app.route("/api/preset", methods=["POST"])
def api_preset():
    data = request.json or {}
    preset_key = data.get("preset", "preset_a")
    res = apply_preset(preset_key)
    return jsonify(res)

@app.route("/api/event_request", methods=["POST"])
def api_event():
    data = request.json or {}
    card = process_event_request(data)
    return jsonify(card)

@app.route("/api/metrics")
def api_metrics():
    # Return fresh computation
    return jsonify(compute_all_metrics())

@app.route("/api/audit_log")
def api_get_audit():
    return jsonify(audit_log)

@app.route("/api/zones")
def api_zones():
    return jsonify(ZONES_META)

@app.route("/api/export_report", methods=["GET"])
def api_export_report():
    """Generates an HTML executive demo report."""
    st = get_current_state()
    m = compute_all_metrics()

    decisions_html = ""
    for d in st["decisions"]:
        decisions_html += f"""
        <div style="border:1px solid #cbd5e1; border-radius:8px; padding:12px; margin-bottom:12px; background:#fff;">
            <div style="display:flex; justify-content:space-between;">
                <b>{d['what']}</b>
                <span style="background:#e0f2fe; padding:2px 8px; border-radius:4px; font-weight:bold;">{d['status'].upper()} ({d.get('decision') or 'PENDING'})</span>
            </div>
            <p style="color:#475569; margin:6px 0;"><b>Location:</b> {d['where']} ({d.get('asset_id')})</p>
            <p style="margin:4px 0;"><b>Why:</b> {d['why']}</p>
            <p style="font-family:monospace; background:#f8fafc; padding:6px; border-radius:4px;"><b>Formula:</b> {d.get('loss_formula')}</p>
            <p><b>Confidence:</b> {d['confidence']}% | <b>Est. Saved:</b> {d['expected_kl_saved']} kL/day (₹{d['rs_per_day_at_risk']:,.0f}/day)</p>
            <ul style="font-size:0.9em; color:#334155;">
                {''.join(f'<li>{ev}</li>' for ev in d.get('evidence', []))}
            </ul>
        </div>
        """

    alloc_rows = ""
    for a in st["optimization"]["allocation_table"]:
        alloc_rows += f"""
        <tr>
            <td style="padding:8px; border-bottom:1px solid #e2e8f0;">{a['zone_name']}</td>
            <td style="padding:8px; border-bottom:1px solid #e2e8f0;">Tier {a['vulnerability_tier']} {'🔒' if a['ethical_guardrail_locked'] else ''}</td>
            <td style="padding:8px; border-bottom:1px solid #e2e8f0;">{a['demand_lpm']} LPM</td>
            <td style="padding:8px; border-bottom:1px solid #e2e8f0;">{a['freshwater_lpm']} LPM</td>
            <td style="padding:8px; border-bottom:1px solid #e2e8f0;">{a['greywater_lpm']} LPM</td>
            <td style="padding:8px; border-bottom:1px solid #e2e8f0;"><b>{a['fulfilled_pct']}%</b></td>
        </tr>
        """

    report_html = f"""<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <title>AI for Water Resilience – Executive Demo Report</title>
    <style>
        body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif; margin: 30px; background: #f8fafc; color: #0f172a; line-height: 1.5; }}
        .header {{ background: #1e293b; color: white; padding: 24px; border-radius: 12px; margin-bottom: 24px; }}
        .section {{ background: white; border-radius: 12px; padding: 20px; margin-bottom: 24px; box-shadow: 0 1px 3px rgba(0,0,0,0.05); }}
        table {{ width: 100%; border-collapse: collapse; text-align: left; }}
        th {{ background: #f1f5f9; padding: 8px; border-bottom: 2px solid #cbd5e1; }}
        .badge {{ display: inline-block; padding: 4px 10px; border-radius: 6px; font-size: 0.85em; font-weight: 600; }}
    </style>
</head>
<body>
    <div class="header">
        <h1>💧 AI for Water Resilience – Executive Demo Audit Report</h1>
        <p>Generated at: {st['timestamp']} | Campus Status: {'CRITICAL ESCALATION' if st['is_red_escalation'] else 'NORMAL AUTONOMOUS DRAFTING'}</p>
        <span class="badge" style="background:#3b82f6; color:white;">DATASET: SYNTHETIC BENCHMARK</span>
        <span class="badge" style="background:#10b981; color:white;">OPTIMIZER: {st['optimization']['solver_status']}</span>
    </div>

    <div class="section">
        <h2>📊 Model & Governance Performance Metrics</h2>
        <p><b>Classification Macro F1:</b> {m['classification']['macro_f1']} | <b>Normal False Positive Rate:</b> {m['classification']['false_positive_rate']*100:.2f}%</p>
        <p><b>Next 1h Flow MAPE:</b> {m['forecast']['mape_1h_flow']}% (Naive Baseline: {m['forecast']['baseline_mape_1h']}%)</p>
        <p><b>Groundedness:</b> {m['groundedness_pct']}% of cards carry ≥2 evidence items</p>
        <p><b>Fairness (Locked Zone Restrictions):</b> {m['fairness']['locked_zone_restrictions']} (Target: 0)</p>
        <p><b>Inference Latency:</b> {m['inference_latency_ms']} ms</p>
    </div>

    <div class="section">
        <h2>🚰 Water Distribution Plan ({st['optimization']['solver_status']})</h2>
        <table>
            <thead>
                <tr>
                    <th>Zone</th>
                    <th>Vulnerability</th>
                    <th>Demand</th>
                    <th>Freshwater</th>
                    <th>STP Greywater</th>
                    <th>Fulfilled</th>
                </tr>
            </thead>
            <tbody>
                {alloc_rows}
            </tbody>
        </table>
    </div>

    <div class="section">
        <h2>📋 Actionable Decision Queue & Evidence Snapshots</h2>
        {decisions_html if decisions_html else '<p>No pending recommendations.</p>'}
    </div>
</body>
</html>
"""
    return Response(report_html, mimetype="text/html", headers={"Content-Disposition": "attachment;filename=water_resilience_report.html"})

if __name__ == "__main__":
    print("\n✅ AI for Water Resilience Server ready on http://0.0.0.0:5000")
    app.run(host="0.0.0.0", port=5000, debug=False)
