# City Sentinel: Strategic Enforcement on Mappls Infrastructure

City Sentinel is a high-integrity operational command center designed to integrate with the **Mappls (MapmyIndia)** ecosystem. Moving beyond standard data visualization, the system leverages Mappls spatial intelligence to bridge the gap between **Violation Patterns** and **Resource Deployment**.

## Core Operational Logic

### 1. Dual-Layer Risk Analysis
The system distinguishes between two critical city narratives:
- **Violation Frequency (Red Map):** Optimized for **Patrol Awareness**. Identifies high-density areas with high ticket volume.
- **Physical Loss (Green Map):** Optimized for **Flow Restoration**. Identifies where illegal parking causes the most physical road capacity loss.

### 2. Explainable Intelligence ("Why this zone?")
The system provides a clear tactical rationale for every recommended zone, answering the commander's question: *Why should I trust this?*
- Reasons are pulled from spatial context (Main road vs Local lane), temporal peaks (Morning/Evening), and growth trends.

### 3. Intervention Simulator
An outcome-focused tool that answers: *What happens if we intervene?*
- Adjust the officer count to see real-time estimates of:
  - **Violations Reduced/hr**
  - **Strategic Zones Covered**
  - **Road Space Recovered (Units)**

---

## Technical Architecture

### Predictive Engine
- **Model:** Random Forest Regressor.
- **Validation:** R² of 0.84, demonstrating high inference stability across city DNA.
- **Features:** H3 Hexagonal Spatial Indexing, Hour of Day, and Day of Week.
- **Methodology:** Data is normalized against historical patterns to distinguish real hotspots from simple enforcement bias.

### Tactical Decision Matrix
Deployment recommendations are based on a tiered SOP-based logic:
- **Immediate Clearing:** High-impact arterial blockages.
- **Citation Wave:** High-frequency local surges.
- **Camera Monitoring:** Strategic automation for moderate risks.

---

## Technical Stack
- **Engine:** Python, Pandas, NumPy
- **Machine Learning:** Scikit-Learn
- **Dashboard:** Streamlit
- **Mapping:** Folium with Branca Color ramps

## How to Run
1. Ensure the `.pkl` models and `h3_metadata.csv` are in the root directory.
2. Install dependencies: `pip install -r requirements.txt`
3. Launch the dashboard: `streamlit run app.py`

---
*Developed for high-stakes municipal enforcement decision support.*
