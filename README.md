# ForgePredict AI 🏎️⚙️
> **Motorsport Telemetry Architecture for Industrial Predictive Maintenance**

ForgePredict AI is a premium, data-rich predictive maintenance dashboard inspired by Formula 1 strategy pit-wall telemetry. It monitors industrial equipment health, predicts failure probability using XGBoost, estimates Remaining Useful Life (RUL), and delivers maintenance strategy recommendations.

---

## 🌟 Key Features

*   **F1 Telemetry Dashboard**: Real-time telemetry dials showing active asset standings, load stress profiles (analogous to F1 tire compounds: Soft, Medium, Hard, Wet), and future sensor forecast trendlines.
*   **Explainable XGBoost Engine**: Transparent diagnostic panel showing XGBoost feature attribution weights to isolate exactly which sensor is causing machine warnings.
*   **Interactive Pit Strategy**: Actively test operating strategies like Eco speed caps, simulate maintenance box windows, and print strategy reports on the fly.
*   **Failure Replay Scrubber**: A timeline scrubber that lets operators freeze the dashboard and rewind telemetry logs up to 10 hours in the past to inspect degradation periods.
*   **AI Chat Assistant**: A rule-based conversational NLP bot connected directly to the active telemetry DataFrame to answer questions like *"Why is Asset-03 risky?"* or *"How many hours before failure?"*.
*   **AI Model Performance Lab**: A dedicated ML evaluation suite showing validation curves (Confusion Matrix heatmap, ROC Curve sensitivity, global SHAP feature importances) and training hyperparameters.
*   **Enterprise PDF Reporting**: Prints clean, formatted strategy reports directly from the browser by auto-hiding interactive buttons and panels using print-media styling.

---

## ⚙️ System Architecture

```mermaid
graph TD
    subgraph "Data Generation (data_generator.py)"
        A[Asset Simulation Engine] -->|Continuous Telemetry| B[Raw Sensors: Vib, Temp, Press, RPM, Flow]
        B -->|Analogy: soft/medium/hard/wet tires| C[Load stress profiles]
    end

    subgraph "Feature Engineering (models.py)"
        C --> D[6-Step Rolling Window Features]
    end

    subgraph "Machine Learning Engine (models.py & forecasting.py)"
        D -->|Feature Vector| E[XGBoost Classifier]
        D -->|Feature Vector| F[XGBoost Regressor]
        D -->|Telemetry Trend| G[Double Exp Smoothing]
        
        E -->|Failure Risk %| H[State Evaluation]
        F -->|RUL Estimate| H
        G -->|10hr Extrapolations| I[Threshold Crossing Estimator]
    end

    subgraph "Decision & Recommendations"
        H --> J[F1 Strategy Engine]
        I --> J
        J -->|Risk < 25%| K["STAY OUT (Normal)"]
        J -->|Risk 25-65%| L["DE-RATE SPEED (Eco)"]
        J -->|Risk > 65% / Fault| M["BOX NOW (Repair Alert)"]
    end

    subgraph "F1 Pit Wall Dashboard (app.py)"
        K --> N[Interactive Dash UI]
        L --> N
        M --> N
        N --> O[Real-Time Standings]
        N --> P[Interactive Telemetry Forecasts]
        N --> Q[Diagnostics Bar Chart]
        N --> R[Factory Floor GPS Map]
        N --> S[Pit Checklist Recommendations]
    end

    style A fill:#0c1322,stroke:#1e293b,stroke-width:2px,color:#fff
    style B fill:#0c1322,stroke:#1e293b,stroke-width:2px,color:#fff
    style C fill:#0c1322,stroke:#1e293b,stroke-width:2px,color:#fff
    style D fill:#111827,stroke:#3b82f6,stroke-width:2px,color:#fff
    style E fill:#111827,stroke:#00f0ff,stroke-width:2px,color:#fff
    style F fill:#111827,stroke:#00f0ff,stroke-width:2px,color:#fff
    style G fill:#111827,stroke:#00f0ff,stroke-width:2px,color:#fff
    style H fill:#111827,stroke:#1e293b,stroke-width:2px,color:#fff
    style I fill:#111827,stroke:#1e293b,stroke-width:2px,color:#fff
    style J fill:#111827,stroke:#ffdf00,stroke-width:2px,color:#fff
    style K fill:#0f172a,stroke:#39ff14,stroke-width:2px,color:#39ff14
    style L fill:#0f172a,stroke:#ffdf00,stroke-width:2px,color:#ffdf00
    style M fill:#0f172a,stroke:#ff073a,stroke-width:2px,color:#ff073a
    style N fill:#0a0e17,stroke:#00f0ff,stroke-width:3px,color:#00f0ff
```

---

## 🛠️ Technology Stack

*   **Core**: Python
*   **Web Dashboard**: Plotly Dash, Dash Bootstrap Components
*   **Visualizations**: Plotly Graph Objects (SVG figures, heatmaps, scatter plots)
*   **Machine Learning**: XGBoost (Classifier & Regressor), Scikit-Learn
*   **Data Wrangling**: Pandas, NumPy
*   **Styling**: Custom CSS (glowing stats, radar-conic conics, typography overlays, responsive panels)

---

## 📂 Project Structure

*   `app.py`: Main dashboard code handling page routing, interactive callbacks, and custom tab views.
*   `models.py`: Trains the XGBoost classifier and regressor on startup.
*   `data_generator.py`: Simulates 4 assets under soft/medium/hard tyre stress profiles.
*   `forecasting.py`: Implements Double Exponential Smoothing for sensor trends.
*   `assets/custom.css`: Cyber-dark motorsport theme stylesheets.
*   `run_dashboard.bat`: Zero-config launcher script.

---

## 🚀 How to Run

### One-Click Setup (Windows)
Double-click the **`run_dashboard.bat`** file in the project folder. The script will automatically:
1. Set up a Python virtual environment (`.venv`) if not already present.
2. Install all dependencies from `requirements.txt` using the fast `uv` installer.
3. Start the dashboard server and retrain the XGBoost models.
4. Launch your browser at [http://127.0.0.1:8050/](http://127.0.0.1:8050/).

---

## ✍️ Author
**Kavyaa Jaiswal** - *AI Engineer*
*   [GitHub](https://github.com/kavyyyaaa)
*   [LinkedIn](https://linkedin.com)
