# AI for Water Resilience – Campus Decision Intelligence Prototype# 💧 AquaGuard AI
### Prescriptive Smart Campus Water Resilience & Ethical Dispatch Co-Pilot

<p align="center">

**TCS AI Hackathon | Challenge 12 | Theme 3: AI for Impact**

*Moving Beyond Passive Water Monitoring to Intelligent, Ethical and Prescriptive Decision-Making.*

</p>

---

## 🌍 1. Problem Statement

Water resilience is a major challenge for residential campuses, cities and industrial facilities.

Traditional water monitoring systems primarily rely on dashboards and threshold-based alerts. However, abnormal water consumption does not always indicate a physical leak. It may result from:

- Legitimate morning and evening demand surges.
- Underground pipeline leakage.
- Faulty IoT sensors and telemetry interruptions.
- Weather-related consumption changes.
- Academic schedules and campus events.
- Inefficient freshwater distribution.
- Poor maintenance prioritization.

These challenges become more critical during municipal supply cuts, summer heatwaves and reservoir shortages.

A monitoring dashboard can identify abnormal readings, but it cannot independently determine the best operational intervention.

**Our objective is to transform water monitoring into a prescriptive decision intelligence system that determines WHAT should be done, WHERE it should be done, and WHY it should be done, while protecting vulnerable populations.**

## 🚀 2. Our Solution – AquaGuard AI

AquaGuard AI is an AI-powered smart campus water resilience platform designed around a closed-loop operational intelligence architecture.

It integrates physics-informed machine learning, demand forecasting, natural language processing, mathematical optimization and human-in-the-loop governance.

The system follows seven operational stages:

```text
MONITOR
   ↓
DETECT
   ↓
PREDICT
   ↓
PRIORITIZE
   ↓
OPTIMIZE
   ↓
INTERVENE
   ↓
MEASURE
   ↓
CONTINUOUS ADAPTATION
```

Unlike conventional monitoring platforms, AquaGuard AI generates evidence-backed intervention recommendations, calculates expected savings and verifies outcomes against counterfactual baselines.

## 🏫 3. Smart Campus – Six Hydraulic Zones

The proposed architecture models an 8,000-student residential campus as a connected hydraulic micro-grid.

The campus is divided into six zones based on operational requirements and population vulnerability.

| Zone | Campus Facility | Vulnerability Tier | AI Resource Policy |
|---|---|---|---|
| C1 | Campus Health Centre & Ward | Tier 1 | 100% potable supply protection |
| C2 | Student Hostels A–D | Tier 1 | 100% potable supply protection |
| C3 | Central Mess & RO Drinking Kiosks | Tier 1 | 100% potable supply protection |
| C4 | Academic Blocks & Wet Labs | Tier 2 | Off-peak scheduling and controlled reuse |
| C5 | Library HVAC Cooling Towers | Tier 3 | Treated STP greywater substitution |
| C6 | Sports Complex & Campus Lawns | Tier 3 | Irrigation optimization and greywater reuse |

### Vulnerability-Based Resource Management

**Tier 1 – Critical and Protected**

Healthcare, residential students and essential food services receive the highest protection.

**Tier 2 – Schedule-Controlled**

Academic and laboratory demand can be optimized during non-operational hours.

**Tier 3 – Flexible and Circular**

HVAC cooling and irrigation are the primary conservation buffers during supply shortages.

## 🧠 4. Seven-Stage AI Architecture

### Stage 1 – MONITOR: Multi-Modal Data Ingestion

AquaGuard AI integrates multiple sources of campus operational information.

**Hydraulic and IoT telemetry:**
- Water flow measurements (L/min).
- Pipeline pressure (PSI).
- Acoustic vibration (dB).
- Overhead Tank (OHT) level and loss percentage.
- Sensor packet jitter and telemetry health.

**External and operational information:**
- Weather forecasts.
- Academic timetable.
- Semester and examination schedules.
- Cultural events and campus activities.
- Student and warden maintenance complaints.

**NLP integration:** Gemini 2.5 Flash converts unstructured maintenance complaints into structured severity and risk information.

### Stage 2 – DETECT: Physics-Informed 4-Way Anomaly Classification

A regularized XGBoost classifier identifies four operational conditions.

| Class | Condition | Diagnostic Evidence |
|---|---|---|
| 0 | Normal | Flow, pressure and mass balance within expected ranges |
| 1 | Physical Pipe Leak | Increased flow, reduced pressure and abnormal hydraulic loss |
| 2 | Demand Spike | Increased consumption with legitimate demand and normal mass balance |
| 3 | Sensor Fault | Telemetry spikes, flatlines or packet jitter inconsistent with physical conditions |

**Reported validation performance:**

| Metric | Result |
|---|---:|
| Validation Accuracy | 98.53% |
| Macro F1 Score | 0.9578 |
| Leak Precision | 0.99 |
| Leak Recall | 0.93 |
| Leak F1 Score | 0.96 |
| Demand Spike F1 | 0.94 |
| Sensor Fault F1 | 0.94 |

The model is evaluated using 6,480 hourly campus records.

### Physics-Based Diagnostic Intelligence

The system combines multiple physical indicators instead of relying on flow thresholds alone.

**Physical leakage:**
- Flow residual increases.
- Downstream pressure decreases.
- Acoustic vibration increases.
- Sump-to-OHT mass-balance loss exceeds the configured threshold.

**Legitimate demand surge:**
- Increased flow during morning rush hours, heatwaves or campus events.
- Mild pressure reduction due to friction.
- OHT mass balance remains consistent with legitimate consumption.

**Sensor recording fault:**
- Abnormal individual sensor readings.
- Flatline or single-tick spikes.
- Packet jitter.
- Contradictions between flow, pressure and acoustic signals.

### Stage 3 – PREDICT: Demand Forecasting & Reservoir Shortage Prediction

AquaGuard AI predicts future water requirements and reservoir depletion.

**24-hour demand forecasting**

- Model: XGBoost Regressor.
- Output: Predicted zone-wise water demand in kL.
- Reported validation MAPE: 2.91%.

**Sump Time-to-Critical (TTC)**

The system estimates how long the main sump can sustain operations before reaching its critical storage threshold.

The critical storage reference is 15% dead storage.

This enables proactive interventions before the campus experiences a severe shortage.

### Stage 4 – PRIORITIZE: Composite Urgency Matrix

Every detected incident receives a deterministic priority score between 0 and 100.

| Priority Factor | Weight |
|---|---:|
| Water Loss | 35% |
| Zone Vulnerability Tier | 30% |
| Pressure Deficit | 20% |
| NLP Complaint Severity | 15% |
| **Total** | **100%** |

The system prioritizes interventions according to water loss, population vulnerability, pressure conditions and complaint severity.

This prevents maintenance teams from relying only on the order in which complaints arrive.

### Stage 5 – OPTIMIZE: Mathematical Distribution Optimization

AquaGuard AI uses constrained Linear Programming through:

`scipy.optimize.linprog(method='highs')`

The optimization engine determines the distribution of potable freshwater and treated STP greywater across six hydraulic zones.

**Decision variables:**

- Freshwater allocation: f₁ ... f₆.
- Treated greywater allocation: g₁ ... g₆.

**Hard ethical constraints:**

For Tier-1 zones C1, C2 and C3:

- Freshwater allocation equals 100% of essential demand.
- Greywater substitution is zero.
- Autonomous freshwater restrictions are prohibited.

**Circular water utilization:**

- Tier 2 supports controlled STP greywater substitution.
- Tier 3 supports higher greywater utilization.
- Non-essential irrigation can be reduced during shortages.

**Demonstration result:**

During the 32% heatwave supply-deficit scenario:

- Tier-1 potable demand remains fully protected.
- 80 kL of treated STP greywater is routed to suitable non-potable applications.
- Sports turf irrigation is curtailed.

### Stage 6 – INTERVENE: Prescriptive What, Where and Why

The system generates actionable intervention recommendations.

Each decision card provides:

- Detected operational condition.
- Affected campus zone.
- Probable physical root cause.
- Recommended intervention.
- Exact infrastructure location.
- Supporting sensor evidence.
- Expected water savings.
- Estimated pumping energy savings.
- Human approval requirement.

Example interventions include:

- Opening a secondary ring-main bypass valve.
- Dispatching a maintenance crew.
- Muting a confirmed sensor false alarm.
- Switching to digital-twin-imputed telemetry.
- Diverting treated STP greywater.
- Cancelling irrigation cycles before expected rainfall.

### Stage 7 – MEASURE: Counterfactual Impact Verification

The final stage measures whether the intervention actually produces the expected benefit.

The system compares:

**Counterfactual Baseline:** Expected water consumption without AI intervention.

**Post-Intervention Consumption:** Actual simulated water draw after the intervention.

The difference is used to estimate:

- Litres of water saved.
- Pumping electricity saved (kWh).
- Financial savings.
- Reduction in water loss.
- Fairness and service-level preservation.

This closes the operational feedback loop and provides measurable evidence of intervention effectiveness.

## 🛡️ 5. Responsible AI & Ethical Governance

AquaGuard AI follows a Human + AI Co-Pilot architecture.

### 5.1 Zero Unapproved Autonomous Valve Control

The AI never directly executes consequential valve commands.

Every intervention is presented for human review through an approval workflow:

**APPROVE | MODIFY | REJECT**

This preserves human authority over critical infrastructure decisions.

### 5.2 Mathematical Vulnerability Protection

Tier-1 zones have hard optimization constraints rather than relying on natural-language instructions.

The optimization solver is mathematically prevented from reducing their protected freshwater allocation.

### 5.3 Human-in-the-Loop Escalation

When available freshwater becomes insufficient for protected Tier-1 demand:

- Autonomous intervention is halted.
- The situation is escalated.
- Authorized human approval is required.

The prototype includes a RED escalation mechanism involving the Dean of Campus Infrastructure and Estate Officer.

### 5.4 Auditability

Operator decisions, intervention history and supporting evidence are recorded through an audit trail in `audit_log.json`.

## ⚡ 6. Dynamic Adaptation – Jury Surprise Scenarios

AquaGuard AI includes three predefined demonstration scenarios and interactive operational controls.

### Scenario 1 – Hostel Pipe Leak + Sensor Recording Fault

**Timestamp:** September 3, 2026 – 15:00

**Detected conditions:**
- Physical pipe leakage in Zone C2 (Student Hostels).
- Sensor recording fault in Zone C5 (Library HVAC).

**Recommended actions:**
- Open the secondary ring-main bypass.
- Dispatch the maintenance crew.
- Avoid interrupting hostel water supply.
- Suppress the faulty sensor alarm and use digital-twin-imputed flow.

**Reported impact:**
- 100,686 litres/day of leakage loss prevented.
- 42.3 kWh pumping energy saved.
- 2.5 hours of unnecessary plumber dispatch avoided.

### Scenario 2 – Exam Week Heatwave + Municipal Supply Cut

**Timestamp:** September 30, 2026 – 14:00

**Detected conditions:**
- Temperature reaches 43.5°C.
- Examination-related demand increases.
- Main sump falls to 30.44%.
- Time-to-Critical is estimated at 3.8 hours.

**Recommended actions:**
- Protect potable supply to C1, C2 and C3.
- Divert 80 kL of treated STP greywater.
- Reduce non-essential freshwater irrigation.
- Recalculate reservoir sustainability.

**Reported impact:**
- 100% Tier-1 potable demand fulfillment.
- Reservoir Time-to-Critical extended from 3.8 to 11.4 hours in the demonstration.

### Scenario 3 – Pre-Monsoon Rainfall Forecast

**Timestamp:** September 8, 2026 – 16:00

**Detected conditions:**
- Forecast rainfall probability reaches 84%.

**Recommended actions:**
- Cancel scheduled freshwater irrigation cycles in Zone C6.
- Reduce booster pump duty cycle by 18%.

**Reported impact:**
- 32,000 litres of potable water conserved.
- 14.5 kWh of pumping electricity saved in the simulated 24-hour intervention.

### Interactive Scenario Controls

The prototype additionally supports:

| Scenario | Adaptive Behaviour |
|---|---|
| Supply Cut | Recalculates available freshwater allocation |
| Crew Shortage | Reprioritizes maintenance dispatch |
| Sensor Outage | Switches to fallback evidence and imputed readings |
| Conflicting Evidence | Requires additional human verification |
| Heatwave | Recalculates demand forecasts |
| Unexpected Leak | Generates a new intervention recommendation |
| Rainfall Forecast | Adjusts irrigation scheduling |

## 📊 7. Dataset & Model Robustness

The anomaly-detection benchmark contains 6,480 hourly records.

Five noise and uncertainty mechanisms are included to improve diagnostic robustness:

1. AR(1) autocorrelated hydraulic drift with φ = 0.65.
2. 32% of physical underground leaks have no student complaints.
3. 7% of normal hours include misleading complaints.
4. Skewed Beta weather forecast uncertainty.
5. 2.2% boundary ambiguity.

**XGBoost regularization configuration:**

| Parameter | Value |
|---|---:|
| max_depth | 6 |
| subsample | 0.8 |
| reg_alpha | 1.0 |
| reg_lambda | 2.0 |

These mechanisms are intended to reduce overfitting and improve robustness against noisy operational telemetry.

## 💰 8. Business Impact & Sustainability

AquaGuard AI is designed to provide measurable financial, operational and environmental benefits.

### Reported Annual Impact – 8,000-Student Campus

| Impact Indicator | Reported Value |
|---|---:|
| Annual Leakage Reduction | 48.5 million litres |
| Non-Revenue Water Reduction | 24% to below 4.5% |
| Annual STP Greywater Reuse | 29.2 million litres |
| Increase in Greywater Utilization | 62% |
| Annual Pumping Energy Savings | 32,600 kWh |
| Estimated CO₂e Avoidance | 26.7 tonnes |
| Vulnerable Population Fairness Index | 1.00 |

These are projected prototype impact figures, not independently verified operational savings from BIT.

### Business Value

**For educational institutions:**
- Lower water procurement costs.
- Reduced maintenance response time.
- Better continuity of essential services.
- Improved sustainability reporting.

**For industrial facilities:**
- Optimized water consumption.
- Improved wastewater reuse.
- Reduced operational losses.
- Better infrastructure resilience.

**For municipal utilities:**
- District-level leak prioritization.
- Protection of hospitals and vulnerable communities.
- Improved distribution efficiency.
- Evidence-based maintenance planning.

## 🏗️ 9. Technology Stack

| Layer | Technology |
|---|---|
| Programming Language | Python |
| Backend | Flask |
| Machine Learning | XGBoost |
| Demand Forecasting | XGBoost Regressor |
| Optimization | SciPy Linear Programming |
| NLP | Gemini 2.5 Flash |
| Data Processing | Pandas, NumPy |
| Optimization Solver | HiGHS |
| Frontend | HTML, CSS, JavaScript |
| Audit Storage | JSON |
| Operational Data | IoT / Sensor Telemetry |
| Fallback Intelligence | Deterministic Physics-Informed NLP |

### Enterprise Resilience

AquaGuard AI separates natural-language interpretation from critical infrastructure control.

Gemini 2.5 Flash is used for unstructured complaint parsing. It does not directly control valves or generate unconstrained actuator commands.

If external cloud or LLM services become unavailable, the application uses a deterministic offline fallback for supported parsing operations and local serialized machine-learning pipelines.

The deck reports inference latency below 15 ms for this fallback architecture.

## ⚙️ 10. Installation & Execution

### Prerequisites

- Python 3.10 or later.
- Git.
- VS Code or another Python IDE.

### Step 1 – Clone the Repository

```bash
git clone https://github.com/A-Mohan-raj/water_resilience.git
```

### Step 2 – Navigate to the Project

```bash
cd water_resilience
```

### Step 3 – Create a Virtual Environment

```bash
python -m venv venv
```

Activate on Windows:

```powershell
venv\Scripts\activate
```

### Step 4 – Install Dependencies

```bash
pip install flask pandas numpy scikit-learn xgboost scipy
```

Configure the Gemini API credentials if the cloud NLP integration is enabled. Keep API keys outside source control.

### Step 5 – Run the Application

```bash
python app.py
```

Open:

```text
http://localhost:5000
```

*Note: These are the documented launch commands and dependency requirements. Confirm the actual repository's requirements and configuration before deployment.*

## 📁 11. Project Structure

The documented application entry point and audit file are:

```text
water_resilience/
│
├── app.py
├── README.md
├── audit_log.json
│
├── model/
│   └── Serialized ML pipelines
│
├── data/
│   └── Campus hydraulic benchmark data
│
└── static/
    └── Frontend assets
```

*The model and data directories illustrate the logical architecture; update this tree to match the actual repository contents.*

## 📈 12. Evaluation & Performance Summary

| Capability | Technology | Reported Result |
|---|---|---|
| Four-Way Diagnosis | Regularized XGBoost | 98.53% validation accuracy |
| Classification Quality | XGBoost | 0.9578 Macro F1 |
| Leak Detection | XGBoost | 0.96 F1 |
| Demand Forecasting | XGBoost Regressor | 2.91% validation MAPE |
| Prioritization | Weighted Urgency Matrix | 0–100 score |
| Distribution Optimization | SciPy HiGHS | Tier-1 hard constraints |
| NLP Complaint Parsing | Gemini 2.5 Flash | Structured severity scores |
| Offline Resilience | Deterministic Fallback | Below 15 ms reported inference latency |
| Intervention Measurement | Counterfactual Model | Water and energy savings |

## 🔮 13. Future Enhancements

- Real-time integration with BIT's authorized water infrastructure sensors.
- GIS-based pipeline and leak localization.
- Advanced acoustic leak detection.
- Integration with actual campus maintenance systems.
- Live reservoir and municipal supply integrations.
- Enhanced time-series drift monitoring and retraining.
- Real-world verification of water and energy savings.
- Multi-campus and industrial water resilience deployment.

## 🚀 14. Scalability & TCS Enterprise Deployment

AquaGuard AI is designed around a modular architecture that can extend beyond educational campuses.

**Horizon 1 – Educational and IT Campuses**

Integrate existing Modbus/BACnet flow meters, ultrasonic OHT sensors and campus helpdesk portals.

**Horizon 2 – Industrial Manufacturing Parks**

Adapt vulnerability tiers to industrial safety showers, cooling systems and Zero Liquid Discharge operations.

**Horizon 3 – Municipal Smart Water Grids**

Protect hospitals and low-income wards while optimizing District Metered Area pressure and distribution.

**Adaptive Model Maintenance**

Monitor rolling seven-day forecasting performance. The proposed system triggers incremental XGBoost retraining when baseline demand drift exceeds 8%.

## 🎯 15. Project Objective

AquaGuard AI demonstrates how artificial intelligence can move beyond prediction and become a prescriptive operational decision-support system.

Its central principle is:

**Don't just detect water loss. Diagnose its cause, determine the best intervention, protect vulnerable populations and measure the actual impact.**

---

### Project Information

**Project Name:** AquaGuard AI

**Full Name:** Prescriptive Smart Campus Water Resilience & Ethical Dispatch Co-Pilot

**Institution:** Bannari Amman Institute of Technology (BIT), Sathyamangalam, Tamil Nadu, India.

**Competition:** TCS AI Hackathon

**Challenge:** Challenge 12 – AI for Water Resilience & Smart Resource Management

**Theme:** Theme 3 – AI for Impact

**Domain:** Sustainability

---

<p align="center">

**AquaGuard AI – Intelligent Water Management. Ethical Resource Allocation. Measurable Sustainability.**

</p>
