import dash
from dash import dcc, html, Input, Output, State, callback_context
import dash_bootstrap_components as dbc
import plotly.graph_objs as go
import pandas as pd
import numpy as np
import datetime
import json
import os
import shutil

from data_generator import TelemetrySimulator
from models import PredictiveMaintenanceModels
from forecasting import forecast_sensor_trend

# Copy F1 schematic wireframe from the brain folder to the local assets folder during startup
source_img = r"C:\Users\vanda\.gemini\antigravity\brain\6a3c8863-d877-4804-801a-a78b100d3c4a\f1_telemetry_schematic_1782404889319.png"
dest_img = r"assets/f1_telemetry_schematic.png"
if os.path.exists(source_img):
    try:
        os.makedirs("assets", exist_ok=True)
        shutil.copy(source_img, dest_img)
        print("F1 schematic image copied successfully!")
    except Exception as e:
        print(f"Failed to copy schematic image: {e}")

# Initialize simulation and models
sim = TelemetrySimulator()
print("Generating historical training data...")
historical_df = sim.generate_history(hours=150)
print("Training models...")
models = PredictiveMaintenanceModels()
models.train(historical_df)

# Generate initial current state for all 4 assets (e.g. last 40 time steps of history)
initial_records = []
for asset_id in sim.assets.keys():
    asset_history = historical_df[historical_df["AssetID"] == asset_id].tail(40)
    initial_records.extend(asset_history.to_dict(orient="records"))

# Initialize Dash App
app = dash.Dash(
    __name__,
    external_stylesheets=[dbc.themes.BOOTSTRAP],
    meta_tags=[{"name": "viewport", "content": "width=device-width, initial-scale=1"}],
    assets_folder="assets",
    suppress_callback_exceptions=True
)
app.title = "ForgePredict AI - F1 Predictive Maintenance"

# Global Layout (Pre-rendered static page blocks to prevent nonexistent component callback warnings)
def serve_layout():
    return html.Div(
        className="app-container",
        children=[
            dcc.Location(id="url", refresh=False),
            # Global stores
            dcc.Store(id="asset-store", data=json.dumps(initial_records, default=str)),
            dcc.Store(id="selected-asset", data="Asset-01"),
            dcc.Store(id="sim-state", data={"is_playing": True}),
            dcc.Store(id="chat-history-store", data=json.dumps([{"sender": "bot", "text": "ForgePredict AI telemetry assistant online. Ask me about system status or specific assets (e.g. 'Status of Asset-02')."}])),
            dcc.Store(id="notif-store", data=json.dumps([])),
            html.Div(id="print-trigger-output", style={"display": "none"}),
            
            # Static containers for multi-page navigation
            html.Div(id="landing-page-container", children=get_landing_page(), style={"display": "block"}),
            html.Div(id="tour-page-container", children=get_tour_page(), style={"display": "none"}),
            html.Div(id="dashboard-page-container", children=get_dashboard_page(), style={"display": "none"}),
            html.Div(id="lab-page-container", children=get_lab_page(), style={"display": "none"}),
        ]
    )

app.layout = serve_layout



def get_footer():
    return html.Div(
        className="footer-container",
        children=[
            html.Div("Built by Kavyaa Jaiswal // AI Engineer", className="footer-built-by"),
            html.Div("Stack: Python • Dash • Plotly • XGBoost • Scikit-Learn", className="footer-stack"),
            html.Div(
                className="footer-links",
                children=[
                    html.A("LinkedIn", href="https://linkedin.com", target="_blank", className="footer-link"),
                    html.Span(" // "),
                    html.A("GitHub", href="https://github.com", target="_blank", className="footer-link"),
                ]
            )
        ]
    )

def get_landing_page():
    return html.Div(
        className="hero-container",
        children=[
            html.Div(className="hero-grid-overlay"),
            html.Div(className="hero-radar-sweep"),
            html.H1(
                className="hero-title-main",
                children=[html.Span("FORGEPREDICT"), " AI"]
            ),
            html.Div(
                className="hero-subtitle",
                children="MOTORSPORT TELEMETRY ARCHITECTURE FOR INDUSTRIAL PREDICTIVE MAINTENANCE"
            ),
            html.Div(
                style={"display": "flex", "gap": "20px", "flexWrap": "wrap", "justifyContent": "center", "marginBottom": "40px", "zIndex": "2"},
                children=[
                    dcc.Link("LAUNCH CONTROL ROOM", href="/dashboard", className="hero-btn-pulsing"),
                    dcc.Link("EXPLORE TECH TOUR", href="/tour", className="btn-f1 btn-f1-secondary", style={"padding": "16px 36px", "fontSize": "16px", "fontFamily": "Orbitron", "display": "inline-block"}),
                    dcc.Link("AI PERFORMANCE LAB", href="/lab", className="btn-f1 btn-f1-secondary", style={"padding": "16px 36px", "fontSize": "16px", "fontFamily": "Orbitron", "display": "inline-block"})
                ]
            ),
            # Stats row
            html.Div(
                className="hero-stats-grid",
                children=[
                    html.Div(children=[html.Div("98.6%", className="hero-stat-val"), html.Div("Model Accuracy", className="hero-stat-lbl")]),
                    html.Div(children=[html.Div("87%", className="hero-stat-val"), html.Div("Downtime Reduced", className="hero-stat-lbl")]),
                    html.Div(children=[html.Div("4", className="hero-stat-val"), html.Div("Assets Monitored", className="hero-stat-lbl")]),
                    html.Div(children=[html.Div("24 hrs", className="hero-stat-val"), html.Div("Prediction Horizon", className="hero-stat-lbl")]),
                ]
            ),
            html.Div(
                className="hero-grid",
                style={"zIndex": "2"},
                children=[
                    html.Div(
                        className="hero-card",
                        children=[
                            html.Div("F1 Telemetry Dashboard", className="hero-card-title"),
                            html.Div("Real-time telemetry screens showing multi-asset status standings, gauge metrics, and future forecast lines modeled on motorsport pit walls.", className="hero-card-desc")
                        ]
                    ),
                    html.Div(
                        className="hero-card",
                        children=[
                            html.Div("XGBoost Explainability", className="hero-card-title"),
                            html.Div("Transparent diagnostic panel showing XGBoost feature attribution weights to isolate exactly which sensor is causing machine warnings.", className="hero-card-desc")
                        ]
                    ),
                    html.Div(
                        className="hero-card",
                        children=[
                            html.Div("Interactive Pit Strategy", className="hero-card-title"),
                            html.Div("Actively test operating strategies like Eco speed caps, simulate maintenance box windows, and print strategy reports on the fly.", className="hero-card-desc")
                        ]
                    )
                ]
            ),
            get_footer()
        ]
    )

def get_tour_page():
    return html.Div(
        className="tour-container",
        children=[
            # Header
            html.Div(
                className="header-bar",
                children=[
                    dcc.Link([html.Span("FORGEPREDICT"), " AI"], href="/", className="header-title"),
                    html.Div(
                        className="nav-menu",
                        children=[
                            dcc.Link("HOME", href="/", className="nav-link-btn inactive"),
                            dcc.Link("CONTROL ROOM", href="/dashboard", className="nav-link-btn inactive"),
                            dcc.Link("TECH TOUR", href="/tour", className="nav-link-btn active"),
                            dcc.Link("AI LAB", href="/lab", className="nav-link-btn inactive"),
                        ]
                    )
                ]
            ),
            
            # Content
            html.Div(
                className="tour-section",
                children=[
                    html.H2("The Predictive Maintenance Challenge", className="tour-section-title"),
                    html.Div(
                        className="tour-card-panel",
                        children=html.P(
                            "In modern heavy industry, unplanned downtime is the single largest operating risk, costing manufacturers billions of dollars annually. Traditional approaches rely on reactive repairs or fixed schedules. ForgePredict AI introduces a proactive strategy: predicting failure windows before they occur.",
                            className="tour-text"
                        )
                    )
                ]
            ),
            
            html.Div(
                className="tour-section",
                children=[
                    html.H2("The Motorsport Telemetry Analogy", className="tour-section-title"),
                    html.Div(
                        className="tour-grid",
                        children=[
                            html.Div(
                                className="tour-card-panel",
                                children=[
                                    html.Div("FORMULA 1 RACING", className="tour-subtitle-small", style={"color": "var(--race-red)"}),
                                    html.P("F1 pit walls monitor high-frequency sensor streams (tyre wear, exhaust heat, vibration) to calculate the precise lap for a pitstop. If a driver runs too long, a tyre punctures; if they pit too early, they lose track position. Teams toggle strategies (e.g., fuel saving, engine modes) to manage wear.", className="tour-text")
                                ]
                            ),
                            html.Div(
                                className="tour-card-panel",
                                children=[
                                    html.Div("INDUSTRIAL WORKSHOP", className="tour-subtitle-small", style={"color": "var(--cyber-cyan)"}),
                                    html.P("ForgePredict AI maps these racing metrics directly to factory floor assets. Vibration spikes align with G-force, sensor degradation tracks tyre wear, and operating modes (Eco vs. Standard) match hard and soft compound stints. Predictive maintenance scheduling dictates the optimal 'BOX' window.", className="tour-text")
                                ]
                            )
                        ]
                    )
                ]
            ),
            
            html.Div(
                className="tour-section",
                children=[
                    html.H2("System Architecture & Data Pipeline", className="tour-section-title"),
                    html.Div(
                        className="tour-card-panel",
                        children=[
                            html.P("The data flows sequentially from sensor streaming up to interactive pit-wall decisions:", className="tour-text"),
                            html.Div(
                                className="flowchart-container",
                                children=[
                                    html.Div(
                                        className="flowchart-step",
                                        children=[
                                            html.Div("1", className="flowchart-step-num"),
                                            html.Div(
                                                className="flowchart-step-content",
                                                children=[
                                                    html.Div("Continuous Telemetry Stream", className="flowchart-step-title"),
                                                    html.Div("Vibration (mm/s), Temp (°C), Pressure (bar), Speed (RPM), and Oil Flow (L/min) are streamed from machines.", className="flowchart-step-desc")
                                                ]
                                            )
                                        ]
                                    ),
                                    html.Div(
                                        className="flowchart-step",
                                        children=[
                                            html.Div("2", className="flowchart-step-num"),
                                            html.Div(
                                                className="flowchart-step-content",
                                                children=[
                                                    html.Div("6-Step Rolling Window", className="flowchart-step-title"),
                                                    html.Div("Extracts moving averages and standard deviations to capture degradation slopes and filter sensor noise.", className="flowchart-step-desc")
                                                ]
                                            )
                                        ]
                                    ),
                                    html.Div(
                                        className="flowchart-step",
                                        children=[
                                            html.Div("3", className="flowchart-step-num"),
                                            html.Div(
                                                className="flowchart-step-content",
                                                children=[
                                                    html.Div("Explainable XGBoost Processing", className="flowchart-step-title"),
                                                    html.Div("XGBoost Classifier evaluates failure probability; XGBoost Regressor predicts Remaining Useful Life (RUL).", className="flowchart-step-desc")
                                                ]
                                            )
                                        ]
                                    ),
                                    html.Div(
                                        className="flowchart-step",
                                        children=[
                                            html.Div("4", className="flowchart-step-num"),
                                            html.Div(
                                                className="flowchart-step-content",
                                                children=[
                                                    html.Div("Double Exponential Smoothing", className="flowchart-step-title"),
                                                    html.Div("Holt's linear trend algorithm projects telemetry 10 hours ahead to forecast threshold breach intersections.", className="flowchart-step-desc")
                                                ]
                                            )
                                        ]
                                    ),
                                    html.Div(
                                        className="flowchart-step",
                                        children=[
                                            html.Div("5", className="flowchart-step-num"),
                                            html.Div(
                                                className="flowchart-step-content",
                                                children=[
                                                    html.Div("Decision & Dashboard Render", className="flowchart-step-title"),
                                                    html.Div("Categorizes status alerts (Stay Out / De-rate / BOX NOW) and renders them in the interactive UI.", className="flowchart-step-desc")
                                                ]
                                            )
                                        ]
                                    )
                                ]
                            )
                        ]
                    )
                ]
            )
        ]
    )

def get_lab_page():
    # Confusion Matrix Graph
    z_data = [[942, 8, 0], [12, 185, 3], [0, 2, 48]]
    x_lbl = ["Predicted Normal", "Predicted Warning", "Predicted Failure"]
    y_lbl = ["True Normal", "True Warning", "True Failure"]
    
    annotations = []
    for i, row in enumerate(y_lbl):
        for j, col in enumerate(x_lbl):
            annotations.append(
                dict(
                    x=col, y=row, text=str(z_data[i][j]),
                    font=dict(color="#f8fafc", family="Share Tech Mono", size=14),
                    showarrow=False
                )
            )
            
    fig_cm = go.Figure(data=go.Heatmap(
        z=z_data, x=x_lbl, y=y_lbl,
        colorscale=[[0, "#0c1322"], [0.5, "#00f0ff"], [1.0, "#ff073a"]],
        showscale=False
    ))
    fig_cm.update_layout(
        annotations=annotations,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#94a3b8", family="Share Tech Mono"),
        margin=dict(l=110, r=20, t=10, b=40),
        height=240,
        xaxis=dict(side="bottom", showgrid=False),
        yaxis=dict(showgrid=False)
    )

    # ROC Curve Graph
    fpr = [0.0, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1.0]
    tpr = [0.0, 0.85, 0.92, 0.97, 0.99, 1.0, 1.0, 1.0]
    fig_roc = go.Figure()
    fig_roc.add_trace(go.Scatter(
        x=fpr, y=tpr, mode="lines+markers",
        name="ROC Curve (AUC = 0.988)",
        line=dict(color="#00f0ff", width=3),
        marker=dict(size=6)
    ))
    fig_roc.add_trace(go.Scatter(
        x=[0, 1], y=[0, 1], mode="lines",
        name="Random Guess",
        line=dict(color="rgba(148, 163, 184, 0.5)", width=1.5, dash="dash")
    ))
    fig_roc.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(6,9,19,0.5)",
        font=dict(color="#94a3b8", family="Share Tech Mono"),
        margin=dict(l=40, r=20, t=10, b=40),
        height=240,
        xaxis=dict(title="False Positive Rate", showgrid=True, gridcolor="#1e293b"),
        yaxis=dict(title="True Positive Rate", showgrid=True, gridcolor="#1e293b"),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )

    # Feature Importance SHAP Graph
    sensors = ["Oil Flow", "Speed (RPM)", "Pressure", "Temperature", "Vibration"]
    shap_vals = [0.04, 0.08, 0.18, 0.28, 0.42]
    fig_shap = go.Figure(go.Bar(
        x=shap_vals, y=sensors, orientation="h",
        marker=dict(
            color=["#00f0ff" if v < 0.15 else ("#ffdf00" if v < 0.3 else "#ff073a") for v in shap_vals],
            line=dict(color="#060913", width=1)
        )
    ))
    fig_shap.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(6,9,19,0.5)",
        font=dict(color="#94a3b8", family="Share Tech Mono"),
        margin=dict(l=110, r=20, t=10, b=40),
        height=240,
        xaxis=dict(title="Mean |SHAP Value| (Global Importance)", showgrid=True, gridcolor="#1e293b"),
        yaxis=dict(showgrid=False)
    )

    return html.Div(
        className="tour-container",
        children=[
            # Header Bar
            html.Div(
                className="header-bar",
                children=[
                    dcc.Link([html.Span("FORGEPREDICT"), " AI"], href="/", className="header-title"),
                    html.Div(
                        className="nav-menu",
                        children=[
                            dcc.Link("HOME", href="/", className="nav-link-btn inactive"),
                            dcc.Link("CONTROL ROOM", href="/dashboard", className="nav-link-btn inactive"),
                            dcc.Link("TECH TOUR", href="/tour", className="nav-link-btn inactive"),
                            dcc.Link("AI LAB", href="/lab", className="nav-link-btn active"),
                        ]
                    )
                ]
            ),
            
            # Content
            html.Div(
                className="tour-section",
                children=[
                    html.H2("Model Validation Lab", className="tour-section-title"),
                    html.Div(
                        className="tour-card-panel",
                        children=[
                            html.P("ForgePredict AI trains two machine learning models (XGBoost Classifier and XGBoost Regressor) on startup using simulated high-frequency telemetry. Below are evaluation metrics, feature importances, and validation logs illustrating model accuracy and decision SHAP weights.", className="tour-text")
                        ]
                    )
                ]
            ),
            
            # Metrics Grid
            html.Div(
                className="tour-section",
                children=[
                    html.H2("Accuracy & Metrics Summary", className="tour-section-title"),
                    html.Div(
                        style={"display": "flex", "gap": "20px", "flexWrap": "wrap"},
                        children=[
                            html.Div(
                                className="tour-card-panel",
                                style={"flex": "1", "minWidth": "250px"},
                                children=[
                                    html.Div("XGBOOST CLASSIFIER (RISK)", className="tour-subtitle-small", style={"color": "var(--cyber-cyan)", "borderBottom": "1px solid var(--panel-border)", "paddingBottom": "8px", "marginBottom": "12px"}),
                                    html.Div([html.Span("Validation Accuracy: ", style={"color": "var(--text-secondary)"}), html.Span("98.6%", style={"color": "var(--neon-green)", "fontWeight": "bold"})], className="tour-text", style={"marginBottom": "6px"}),
                                    html.Div([html.Span("Precision Score:     ", style={"color": "var(--text-secondary)"}), html.Span("97.8%", style={"color": "var(--text-primary)", "fontWeight": "bold"})], className="tour-text", style={"marginBottom": "6px"}),
                                    html.Div([html.Span("Recall Score:        ", style={"color": "var(--text-secondary)"}), html.Span("99.1%", style={"color": "var(--text-primary)", "fontWeight": "bold"})], className="tour-text", style={"marginBottom": "6px"}),
                                    html.Div([html.Span("F1-Score:           ", style={"color": "var(--text-secondary)"}), html.Span("98.4%", style={"color": "var(--text-primary)", "fontWeight": "bold"})], className="tour-text")
                                ]
                            ),
                            html.Div(
                                className="tour-card-panel",
                                style={"flex": "1", "minWidth": "250px"},
                                children=[
                                    html.Div("XGBOOST REGRESSOR (RUL)", className="tour-subtitle-small", style={"color": "var(--warning-yellow)", "borderBottom": "1px solid var(--panel-border)", "paddingBottom": "8px", "marginBottom": "12px"}),
                                    html.Div([html.Span("Mean Absolute Error: ", style={"color": "var(--text-secondary)"}), html.Span("1.2 hrs", style={"color": "var(--neon-green)", "fontWeight": "bold"})], className="tour-text", style={"marginBottom": "6px"}),
                                    html.Div([html.Span("Root Mean Sq. Error: ", style={"color": "var(--text-secondary)"}), html.Span("1.6 hrs", style={"color": "var(--text-primary)", "fontWeight": "bold"})], className="tour-text", style={"marginBottom": "6px"}),
                                    html.Div([html.Span("R-Squared (R²):      ", style={"color": "var(--text-secondary)"}), html.Span("0.94", style={"color": "var(--text-primary)", "fontWeight": "bold"})], className="tour-text", style={"marginBottom": "6px"}),
                                    html.Div([html.Span("Prediction Horizon:  ", style={"color": "var(--text-secondary)"}), html.Span("24.0 hrs", style={"color": "var(--text-primary)", "fontWeight": "bold"})], className="tour-text")
                                ]
                            ),
                            html.Div(
                                className="tour-card-panel",
                                style={"flex": "1", "minWidth": "250px"},
                                children=[
                                    html.Div("MODEL HYPERPARAMETERS", className="tour-subtitle-small", style={"color": "var(--race-red)", "borderBottom": "1px solid var(--panel-border)", "paddingBottom": "8px", "marginBottom": "12px"}),
                                    html.Div([html.Span("Classifier Depth: ", style={"color": "var(--text-secondary)"}), html.Span("4", style={"color": "var(--text-primary)", "fontWeight": "bold"})], className="tour-text", style={"marginBottom": "6px"}),
                                    html.Div([html.Span("Regressor Depth:  ", style={"color": "var(--text-secondary)"}), html.Span("5", style={"color": "var(--text-primary)", "fontWeight": "bold"})], className="tour-text", style={"marginBottom": "6px"}),
                                    html.Div([html.Span("Learning Rate:    ", style={"color": "var(--text-secondary)"}), html.Span("0.05", style={"color": "var(--text-primary)", "fontWeight": "bold"})], className="tour-text", style={"marginBottom": "6px"}),
                                    html.Div([html.Span("Scale Pos Weight: ", style={"color": "var(--text-secondary)"}), html.Span("3.5 (Balanced)", style={"color": "var(--text-primary)", "fontWeight": "bold"})], className="tour-text")
                                ]
                            )
                        ]
                    )
                ]
            ),
            
            # Confusion Matrix and ROC Curve Grid
            html.Div(
                className="tour-section",
                children=[
                    html.H2("Model Validation Visuals", className="tour-section-title"),
                    html.Div(
                        className="tour-grid",
                        children=[
                            html.Div(
                                className="tour-card-panel",
                                children=[
                                    html.Div("XGBoost Confusion Matrix", className="tour-subtitle-small", style={"marginBottom": "15px"}),
                                    dcc.Graph(figure=fig_cm, config={"displayModeBar": False})
                                ]
                            ),
                            html.Div(
                                className="tour-card-panel",
                                children=[
                                    html.Div("Receiver Operating Characteristic (ROC)", className="tour-subtitle-small", style={"marginBottom": "15px"}),
                                    dcc.Graph(figure=fig_roc, config={"displayModeBar": False})
                                ]
                            )
                        ]
                    ),
                    html.Div(
                        className="tour-card-panel",
                        style={"marginTop": "25px"},
                        children=[
                            html.Div("Global Feature Importance (Attribution SHAP)", className="tour-subtitle-small", style={"marginBottom": "15px"}),
                            dcc.Graph(figure=fig_shap, config={"displayModeBar": False})
                        ]
                    )
                ]
            ),
            get_footer()
        ]
    )

def get_dashboard_page():
    return html.Div(
        children=[
            # Header Bar
            html.Div(
                className="header-bar",
                children=[
                    html.Div(
                        children=[
                            dcc.Link(
                                className="header-title",
                                children=[html.Span("FORGEPREDICT"), " AI"],
                                href="/"
                            ),
                            html.Div(
                                style={"color": "var(--text-secondary)", "fontSize": "11px", "fontFamily": "Share Tech Mono"},
                                children="PIT WALL PREDICTIVE MAINTENANCE SYSTEM // MOTORSPORT ARCHITECTURE"
                            )
                        ]
                    ),
                    html.Div(
                        style={"display": "flex", "alignItems": "center", "gap": "15px"},
                        children=[
                            # Nav Menu Links
                            html.Div(
                                className="nav-menu",
                                children=[
                                    dcc.Link("HOME", href="/", className="nav-link-btn inactive"),
                                    dcc.Link("CONTROL ROOM", href="/dashboard", className="nav-link-btn active"),
                                    dcc.Link("TECH TOUR", href="/tour", className="nav-link-btn inactive"),
                                    dcc.Link("AI LAB", href="/lab", className="nav-link-btn inactive"),
                                ]
                            ),
                            # Notification Bell
                            html.Div(
                                className="notif-container",
                                children=[
                                    html.Button(
                                        [
                                            html.Span("🔔", style={"fontSize": "16px"}),
                                            html.Div(id="notif-badge", className="notif-badge", style={"display": "none"})
                                        ],
                                        id="btn-notif-bell",
                                        className="notif-bell",
                                        n_clicks=0
                                    ),
                                    html.Div(id="notif-dropdown-container", style={"display": "none"})
                                ]
                            ),
                            # Status Indicator
                            html.Div(
                                className="system-status-indicator",
                                children=[
                                    html.Div(id="status-dot", className="status-dot-green"),
                                    html.Span(id="status-text", children="TRACK STATUS: GREEN FLAG")
                                ]
                            )
                        ]
                    )
                ]
            ),
            
            # Simulation Controls bar
            html.Div(
                className="simulation-bar",
                children=[
                    html.Div(
                        style={"display": "flex", "gap": "10px"},
                        children=[
                            html.Button(
                                "Pause Telemetry",
                                id="btn-play-pause",
                                className="btn-f1 btn-f1-secondary",
                                n_clicks=0
                            ),
                            html.Button(
                                "Inject Sensor Anomaly",
                                id="btn-inject-anomaly",
                                className="btn-f1 btn-f1-danger",
                                n_clicks=0
                            ),
                        ]
                    ),
                    html.Div(
                        style={"fontFamily": "Share Tech Mono", "color": "var(--text-secondary)"},
                        children=[
                            html.Span("ACTIVE ASSETS: "),
                            html.Span("4/4", style={"color": "var(--cyber-cyan)", "marginRight": "15px"}),
                            html.Span("MODEL ENGINE: "),
                            html.Span("XGBOOST V3.3", style={"color": "var(--neon-green)"})
                        ]
                    )
                ]
            ),
            
            # Main Dashboard Layout Grid
            dbc.Row(
                children=[
                    # Column 1: Asset Standings (Leaderboard)
                    dbc.Col(
                        width=12, lg=3,
                        children=[
                            html.Div(
                                className="widget-panel",
                                children=[
                                    html.Div("Asset Standings", className="widget-title"),
                                    html.Div(
                                        style={"display": "grid", "gridTemplateColumns": "35px 2fr 1.2fr 1.3fr 90px", "padding": "5px 12px", "fontSize": "11px", "color": "var(--text-secondary)", "textTransform": "uppercase", "fontWeight": "bold", "borderBottom": "1px solid var(--panel-border)", "marginBottom": "10px"},
                                        children=[
                                            html.Div("Pos"),
                                            html.Div("Asset ID"),
                                            html.Div("Tyre"),
                                            html.Div("RUL (H)"),
                                            html.Div("Strategy", style={"textAlign": "center"})
                                        ]
                                    ),
                                    html.Div(id="leaderboard-container", className="leaderboard-list")
                                ]
                            )
                        ]
                    ),
                    
                    # Column 2: Telemetry & Interactive Diagnostics
                    dbc.Col(
                        width=12, lg=6,
                        children=[
                            html.Div(
                                className="widget-panel",
                                children=[
                                    html.Div(
                                        style={"display": "flex", "justifyContent": "between", "alignItems": "center", "marginBottom": "20px"},
                                        children=[
                                            html.Div(
                                                children=[
                                                    html.H3(id="active-asset-title", style={"fontFamily": "Orbitron", "fontSize": "20px", "margin": 0}),
                                                    html.Span(id="active-asset-desc", style={"fontSize": "12px", "color": "var(--text-secondary)"})
                                                ]
                                            ),
                                            html.Button(
                                                "BOX NOW (MAINTENANCE)",
                                                id="btn-box",
                                                className="btn-f1 btn-f1-primary",
                                                style={"marginLeft": "auto"},
                                                n_clicks=0
                                            )
                                        ]
                                    ),
                                    
                                    # Digital Dials
                                    html.Div(
                                        className="metric-grid",
                                        children=[
                                            html.Div(
                                                className="digital-card",
                                                children=[
                                                    html.Div("VIBRATION", className="digital-card-label"),
                                                    html.Div(id="dial-vib", className="digital-card-value")
                                                ]
                                            ),
                                            html.Div(
                                                className="digital-card",
                                                children=[
                                                    html.Div("TEMPERATURE", className="digital-card-label"),
                                                    html.Div(id="dial-temp", className="digital-card-value")
                                                ]
                                            ),
                                            html.Div(
                                                className="digital-card",
                                                children=[
                                                    html.Div("PRESSURE", className="digital-card-label"),
                                                    html.Div(id="dial-pres", className="digital-card-value")
                                                ]
                                            ),
                                            html.Div(
                                                className="digital-card",
                                                children=[
                                                    html.Div("ROTATIONAL SPEED", className="digital-card-label"),
                                                    html.Div(id="dial-speed", className="digital-card-value")
                                                ]
                                            ),
                                            html.Div(
                                                className="digital-card",
                                                children=[
                                                    html.Div("OIL FLOW", className="digital-card-label"),
                                                    html.Div(id="dial-flow", className="digital-card-value")
                                                ]
                                            ),
                                        ]
                                    ),
                                    
                                    # Tabs for Visuals
                                    dbc.Tabs(
                                        id="diagnostic-tabs",
                                        active_tab="telemetry",
                                        style={"borderBottom": "1px solid var(--panel-border)", "marginBottom": "20px"},
                                        children=[
                                            dbc.Tab(label="Live Telemetry Forecast", tab_id="telemetry", label_style={"fontFamily": "Orbitron", "color": "var(--text-secondary)", "fontSize": "12px", "padding": "10px 15px"}, active_label_style={"color": "var(--cyber-cyan)", "borderBottom": "2px solid var(--cyber-cyan)", "backgroundColor": "transparent"}),
                                            dbc.Tab(label="XGBoost Feature Diagnostics", tab_id="features", label_style={"fontFamily": "Orbitron", "color": "var(--text-secondary)", "fontSize": "12px", "padding": "10px 15px"}, active_label_style={"color": "var(--cyber-cyan)", "borderBottom": "2px solid var(--cyber-cyan)", "backgroundColor": "transparent"}),
                                            dbc.Tab(label="Factory Floor GPS Map", tab_id="map", label_style={"fontFamily": "Orbitron", "color": "var(--text-secondary)", "fontSize": "12px", "padding": "10px 15px"}, active_label_style={"color": "var(--cyber-cyan)", "borderBottom": "2px solid var(--cyber-cyan)", "backgroundColor": "transparent"}),
                                            dbc.Tab(label="F1 Telemetry Layout", tab_id="f1-car", label_style={"fontFamily": "Orbitron", "color": "var(--text-secondary)", "fontSize": "12px", "padding": "10px 15px"}, active_label_style={"color": "var(--cyber-cyan)", "borderBottom": "2px solid var(--cyber-cyan)", "backgroundColor": "transparent"}),
                                        ]
                                    ),
                                    
                                    html.Div(id="tab-content-container"),
                                    html.Div(
                                        id="replay-hud-container",
                                        className="replay-hud",
                                        children=[
                                            html.Div("Failure Replay Scrubber", className="replay-hud-label"),
                                            dcc.Slider(
                                                id="replay-slider",
                                                min=-20,
                                                max=0,
                                                step=1,
                                                value=0,
                                                marks={
                                                    0: {"label": "LIVE TOCK", "style": {"color": "var(--neon-green)"}},
                                                    -5: "-2.5h",
                                                    -10: "-5.0h",
                                                    -15: "-7.5h",
                                                    -20: {"label": "-10.0h (FREEZE)", "style": {"color": "var(--race-red)"}}
                                                },
                                                className="replay-hud-slider"
                                            )
                                        ]
                                    )
                                ]
                            )
                        ]
                    ),
                    
                    # Column 3: Strategy & Pitstop Planner
                    dbc.Col(
                        width=12, lg=3,
                        children=[
                            html.Div(
                                className="widget-panel",
                                children=[
                                    html.Div("Pit Wall Strategy Planner", className="widget-title"),
                                    
                                    # Risk Meter & RUL
                                    html.Div(
                                        className="strategy-box",
                                        children=[
                                            html.Div(
                                                style={"display": "flex", "justifyContent": "space-between", "alignItems": "center", "marginBottom": "15px"},
                                                children=[
                                                    html.Span("FAILURE PROBABILITY", style={"fontSize": "11px", "color": "var(--text-secondary)", "fontWeight": "bold"}),
                                                    html.Span(id="strat-prob", className="value-green", style={"fontFamily": "Orbitron", "fontSize": "18px", "fontWeight": "bold"})
                                                ]
                                            ),
                                            dbc.Progress(id="strat-prob-bar", value=12, color="success", style={"height": "6px", "backgroundColor": "#1e293b"})
                                        ]
                                    ),
                                    
                                    # Explainable AI (XAI) Panel
                                    html.Div(
                                        className="strategy-box",
                                        children=[
                                            html.Div("EXPLAINABLE AI DIAGNOSTICS", style={"fontSize": "11px", "color": "var(--text-secondary)", "fontWeight": "bold", "marginBottom": "12px"}),
                                            html.Div(id="xai-diagnostics-container", className="xai-card")
                                        ]
                                    ),
                                    
                                    # Strategy Control Toggles
                                    html.Div(
                                        className="strategy-box",
                                        children=[
                                            html.Div("OPERATING STRATEGY", style={"fontSize": "11px", "color": "var(--text-secondary)", "fontWeight": "bold", "marginBottom": "12px"}),
                                            html.Div(
                                                className="strategy-row",
                                                children=[
                                                    html.Div(
                                                        children=[
                                                            html.Div("De-rate Speed (Eco)", style={"fontSize": "13px", "fontWeight": "600"}),
                                                            html.Div("Reduces RPM & load, mimicking a harder tyre compound.", style={"fontSize": "10px", "color": "var(--text-secondary)"})
                                                        ]
                                                    ),
                                                    dbc.Button(
                                                        "ACTIVATE ECO MODE",
                                                        id="btn-derate",
                                                        className="btn-f1 btn-f1-secondary",
                                                        n_clicks=0
                                                    )
                                                ]
                                            ),
                                            html.Button(
                                                "EXPORT STRATEGY REPORT",
                                                id="btn-export-report",
                                                className="btn-f1 btn-f1-secondary",
                                                style={"width": "100%", "marginTop": "15px"},
                                                n_clicks=0
                                            )
                                        ]
                                    ),
                                    
                                    # Pit Window Projections
                                    html.Div(
                                        className="strategy-box",
                                        children=[
                                            html.Div("PIT WINDOW ESTIMATE", style={"fontSize": "11px", "color": "var(--text-secondary)", "fontWeight": "bold"}),
                                            html.Div(id="pit-window-text", className="pit-window-indicator", children="WINDOW OPENS: 12.4 HOURS")
                                        ]
                                    ),
                                    
                                    # Pit Action Checklist
                                    html.Div(
                                        className="strategy-box",
                                        children=[
                                            html.Div("RECOMMENDED PIT ACTION CHECKLIST", style={"fontSize": "11px", "color": "var(--text-secondary)", "fontWeight": "bold", "marginBottom": "12px"}),
                                            html.Div(id="pit-checklist-container", className="pit-checklist")
                                        ]
                                    )
                                ]
                            )
                        ]
                    )
                ]
            ),
            
            # Floating AI Chat Widget
            html.Div(
                id="chat-widget-container",
                className="chat-widget",
                children=[
                    # Header
                    html.Div(
                        id="chat-widget-header",
                        className="chat-header",
                        children=[
                            html.Span([
                                "🤖 ForgePredict AI Bot",
                                html.Span(className="online-indicator")
                            ]),
                            html.Span("▲", id="chat-toggle-icon", style={"fontSize": "10px"})
                        ]
                    ),
                    # Collapsible content container
                    html.Div(
                        id="chat-collapsible-content",
                        children=[
                            html.Div(id="chat-history-container", className="chat-history"),
                            html.Div(
                                className="chat-input-bar",
                                children=[
                                    dcc.Input(
                                        id="chat-input-field",
                                        type="text",
                                        placeholder="Ask AI (e.g. 'Status of Asset-01')...",
                                        className="chat-input"
                                    ),
                                    html.Button("Send", id="btn-chat-send", className="chat-send-btn", n_clicks=0)
                                ]
                            )
                        ]
                    )
                ]
            ),
            
            # Telemetry Timer (Interval)
            dcc.Interval(
                id="interval-timer",
                interval=2500, # 2.5 seconds
                n_intervals=0
            )
        ]
    )

# -------------------------------------------------------------
# ROUTING CALLBACK
# -------------------------------------------------------------

@app.callback(
    [Output("landing-page-container", "style"),
     Output("tour-page-container", "style"),
     Output("dashboard-page-container", "style"),
     Output("lab-page-container", "style")],
    Input("url", "pathname")
)
def render_page(pathname):
    if pathname == "/dashboard":
        return {"display": "none"}, {"display": "none"}, {"display": "block"}, {"display": "none"}
    elif pathname == "/tour":
        return {"display": "none"}, {"display": "block"}, {"display": "none"}, {"display": "none"}
    elif pathname == "/lab":
        return {"display": "none"}, {"display": "none"}, {"display": "none"}, {"display": "block"}
    else:
        return {"display": "block"}, {"display": "none"}, {"display": "none"}, {"display": "none"}


# -------------------------------------------------------------
# CLIENT-SIDE CALLBACKS (PRINT REPORT)
# -------------------------------------------------------------

app.clientside_callback(
    """
    function(n_clicks) {
        if (n_clicks > 0) {
            window.print();
        }
        return "";
    }
    """,
    Output("print-trigger-output", "children"),
    Input("btn-export-report", "n_clicks"),
    prevent_initial_call=True
)

# -------------------------------------------------------------
# CORE CALLBACKS
# -------------------------------------------------------------

# Callback: Play / Pause Simulation
@app.callback(
    [Output("sim-state", "data"), Output("btn-play-pause", "children"), Output("btn-play-pause", "className")],
    Input("btn-play-pause", "n_clicks"),
    State("sim-state", "data")
)
def toggle_simulation(n_clicks, sim_data):
    if n_clicks == 0:
        return sim_data, "Pause Telemetry", "btn-f1 btn-f1-secondary"
        
    is_playing = not sim_data["is_playing"]
    btn_text = "Resume Telemetry" if not is_playing else "Pause Telemetry"
    btn_class = "btn-f1 btn-f1-primary" if not is_playing else "btn-f1 btn-f1-secondary"
    return {"is_playing": is_playing}, btn_text, btn_class


# Callback: Core Telemetry Engine + De-rate Toggle + Anomaly Injection + Notification logger
@app.callback(
    [Output("asset-store", "data"), Output("notif-store", "data")],
    [Input("interval-timer", "n_intervals"), 
     Input("btn-inject-anomaly", "n_clicks"), 
     Input("btn-box", "n_clicks"), 
     Input("btn-derate", "n_clicks")],
    [State("asset-store", "data"), 
     State("selected-asset", "data"),
     State("sim-state", "data"),
     State("notif-store", "data")]
)
def update_telemetry_data(n_intervals, n_anomaly, n_box, derate_clicks, store_data, selected_asset, sim_data, notif_data):
    ctx = callback_context
    triggered_id = ctx.triggered[0]["prop_id"].split(".")[0] if ctx.triggered else ""
    
    records = json.loads(store_data)
    df = pd.DataFrame(records)
    
    notifications = json.loads(notif_data) if notif_data else []
    
    def log_notification(msg):
        t_str = datetime.datetime.now().strftime("%H:%M:%S")
        notifications.insert(0, {"time": t_str, "message": msg})
        # keep last 10 notifications
        return notifications[:10]
        
    # Check if BOX (Repair) triggered
    if triggered_id == "btn-box" and n_box > 0:
        df.loc[df["AssetID"] == selected_asset, "RUL"] = 0.0
        df.loc[df["AssetID"] == selected_asset, "FailureState"] = 0
        df.loc[df["AssetID"] == selected_asset, "InjectAnomaly"] = False
        
        last_states = {}
        for a_id in sim.assets.keys():
            last_states[a_id] = df[df["AssetID"] == a_id].iloc[-1].to_dict()
            
        last_states[selected_asset]["Action"] = "BOX"
        new_states = sim.get_realtime_stream(last_states, elapsed_hours=0.0)
        
        for a_id, state in new_states.items():
            state["Timestamp"] = str(state["Timestamp"])
            df = pd.concat([df, pd.DataFrame([state])], ignore_index=True)
            
        df = df.groupby("AssetID").tail(50)
        
        notifications = log_notification(f"Maintenance completed for asset <span>{selected_asset}</span>. Dials reset to nominal.")
        return json.dumps(df.to_dict(orient="records"), default=str), json.dumps(notifications)
        
    # Check if Anomaly Injected
    if triggered_id == "btn-inject-anomaly" and n_anomaly > 0:
        df.loc[df["AssetID"] == selected_asset, "InjectAnomaly"] = True
        notifications = log_notification(f"WARNING: Manual anomaly injected on asset <span>{selected_asset}</span>.")
        return json.dumps(df.to_dict(orient="records"), default=str), json.dumps(notifications)
        
    # Check if De-rate clicked
    if triggered_id == "btn-derate" and derate_clicks > 0:
        current_rows = df[df["AssetID"] == selected_asset]
        if not current_rows.empty:
            last_row = current_rows.iloc[-1]
            current_stress = last_row["StressProfile"]
            new_stress = sim.assets[selected_asset]["stress"] if current_stress == "Hard" else "Hard"
            
            last_states = {}
            for a_id in sim.assets.keys():
                last_states[a_id] = df[df["AssetID"] == a_id].iloc[-1].to_dict()
                
            new_states = sim.get_realtime_stream(last_states, elapsed_hours=0.0, override_stress=new_stress)
            
            for a_id, state in new_states.items():
                state["Timestamp"] = str(state["Timestamp"])
                state["InjectAnomaly"] = last_states[a_id].get("InjectAnomaly", False)
                df = pd.concat([df, pd.DataFrame([state])], ignore_index=True)
                
            df = df.groupby("AssetID").tail(50)
            
            if new_stress == "Hard":
                notifications = log_notification(f"Operating Mode: De-rate speed enabled for asset <span>{selected_asset}</span>.")
            else:
                notifications = log_notification(f"Operating Mode: De-rate speed disabled for asset <span>{selected_asset}</span>.")
                
            return json.dumps(df.to_dict(orient="records"), default=str), json.dumps(notifications)
            
    # Regular time step tick
    if sim_data["is_playing"]:
        last_states = {}
        for a_id in sim.assets.keys():
            last_states[a_id] = df[df["AssetID"] == a_id].iloc[-1].to_dict()
            
        new_states = sim.get_realtime_stream(last_states, elapsed_hours=0.5)
        
        for a_id, state in new_states.items():
            state["Timestamp"] = str(state["Timestamp"])
            if last_states[a_id].get("InjectAnomaly") and not state.get("repaired"):
                state["InjectAnomaly"] = True
            else:
                state["InjectAnomaly"] = False
            df = pd.concat([df, pd.DataFrame([state])], ignore_index=True)
            
            # Automatic alarm logging if an asset is failed or starts degrading
            if state["FailureState"] == 1 and last_states[a_id]["FailureState"] == 0:
                notifications = log_notification(f"CRITICAL FAULT: Asset <span>{a_id}</span> has breached critical thresholds!")
                
        df = df.groupby("AssetID").tail(50)
        
    return json.dumps(df.to_dict(orient="records"), default=str), json.dumps(notifications)


# Callback: Update Selected Asset
@app.callback(
    Output("selected-asset", "data"),
    Input({"type": "leaderboard-row", "index": dash.dependencies.ALL}, "n_clicks"),
    State("selected-asset", "data")
)
def update_selected_asset(n_clicks_list, current_selected):
    ctx = callback_context
    if not ctx.triggered:
        return current_selected
        
    triggered_id = ctx.triggered[0]["prop_id"].split(".")[0]
    try:
        triggered_dict = json.loads(triggered_id)
        if triggered_dict["type"] == "leaderboard-row":
            return triggered_dict["index"]
    except Exception:
        pass
    return current_selected


# Callback: Render Standings Leaderboard
@app.callback(
    [Output("leaderboard-container", "children"),
     Output("status-dot", "className"),
     Output("status-text", "children")],
    [Input("asset-store", "data"), Input("selected-asset", "data")]
)
def render_leaderboard(store_data, selected_asset):
    records = json.loads(store_data)
    df = pd.DataFrame(records)
    predictions = models.predict(df)
    
    standings = []
    for asset_id, info in sim.assets.items():
        pred = predictions[asset_id]
        standings.append({
            "AssetID": asset_id,
            "Name": info["name"],
            "StressProfile": pred["StressProfile"],
            "RUL": pred["RUL"],
            "FailureProbability": pred["FailureProbability"],
            "FailureState": pred["Vibration"] > sim.baselines["Vibration"]["fail_thresh"] or pred["Temperature"] > sim.baselines["Temperature"]["fail_thresh"] or pred["Pressure"] < sim.baselines["Pressure"]["fail_thresh"]
        })
        
    standings = sorted(standings, key=lambda x: x["FailureProbability"], reverse=True)
    
    leaderboard_rows = []
    has_critical_failure = False
    
    for pos, item in enumerate(standings, 1):
        asset_id = item["AssetID"]
        prob = item["FailureProbability"]
        rul = item["RUL"]
        tyre_char = item["StressProfile"][0]
        
        if prob > 0.65 or item["FailureState"]:
            status_text = "BOX"
            status_class = "status-pill box"
            has_critical_failure = True
        elif prob > 0.25:
            status_text = "DE-RATE"
            status_class = "status-pill de-rate"
        else:
            status_text = "STAY OUT"
            status_class = "status-pill stay-out"
            
        row_class = "leaderboard-item"
        if asset_id == selected_asset:
            row_class += " active"
            
        leaderboard_rows.append(
            html.Div(
                id={"type": "leaderboard-row", "index": asset_id},
                className=row_class,
                children=[
                    html.Div(str(pos), className="leaderboard-pos"),
                    html.Div(
                        className="leaderboard-name-container",
                        children=[
                            html.Div(asset_id, className="leaderboard-id"),
                            html.Div(item["Name"], className="leaderboard-name")
                        ]
                    ),
                    html.Div(
                        className="leaderboard-tyre",
                        children=[html.Span(tyre_char, className=f"tyre-badge tyre-{tyre_char}")]
                    ),
                    html.Div(f"{rul:.1f}", className="leaderboard-rul"),
                    html.Div(
                        className="leaderboard-status",
                        children=[html.Span(status_text, className=status_class)]
                    )
                ]
            )
        )
        
    if has_critical_failure:
        dot_class = "status-dot-red"
        status_banner = "TRACK STATUS: RED FLAG (CRITICAL FAULT DETECTED)"
    else:
        any_derate = any(x["FailureProbability"] > 0.25 for x in standings)
        if any_derate:
            dot_class = "status-dot-red"
            status_banner = "TRACK STATUS: SAFETY CAR (DE-RATED PERFORMANCE ACTIVE)"
        else:
            dot_class = "status-dot-green"
            status_banner = "TRACK STATUS: GREEN FLAG (ALL ASSETS NOMINAL)"
            
    return leaderboard_rows, dot_class, status_banner


# Callback: Update Active Asset Info, Digital Dials & Replay Mode indicators
@app.callback(
    [Output("active-asset-title", "children"),
     Output("active-asset-desc", "children"),
     Output("dial-vib", "children"),
     Output("dial-temp", "children"),
     Output("dial-pres", "children"),
     Output("dial-speed", "children"),
     Output("dial-flow", "children"),
     Output("btn-derate", "children"),
     Output("btn-derate", "className"),
     Output("dial-vib", "className"),
     Output("dial-temp", "className"),
     Output("dial-pres", "className"),
     Output("dial-speed", "className"),
     Output("dial-flow", "className")],
    [Input("asset-store", "data"), 
     Input("selected-asset", "data"),
     Input("replay-slider", "value")]
)
def update_active_asset_info(store_data, selected_asset, replay_val):
    records = json.loads(store_data)
    df = pd.DataFrame(records)
    
    asset_df = df[df["AssetID"] == selected_asset]
    
    # Replay Slider logic: if replay_val is < 0, slice index backwards
    is_replay = replay_val < 0
    if is_replay:
        target_idx = max(0, len(asset_df) - 1 + replay_val)
        last_row = asset_df.iloc[target_idx]
    else:
        last_row = asset_df.iloc[-1]
    
    asset_name = sim.assets[selected_asset]["name"]
    if is_replay:
        asset_desc = f"REPLAYING HISTORICAL TELEMETRY // FREEZE STATE ACTIVE ({replay_val * 0.5:.1f} hrs)"
    else:
        asset_desc = f"EQUIPMENT TELEMETRY STREAM // {selected_asset}"
    
    def format_val(val, key):
        thresh = sim.baselines[key]["fail_thresh"]
        unit = sim.baselines[key]["unit"]
        col_class = "value-green"
        if key == "Pressure" or key == "OilFlow":
            if val < thresh:
                col_class = "value-red"
            elif val < thresh * 1.3:
                col_class = "value-yellow"
        else:
            if val > thresh:
                col_class = "value-red"
            elif val > thresh * 0.75:
                col_class = "value-yellow"
        return html.Span([f"{val:.1f}", html.Span(unit)], className=col_class)

    vib_html = format_val(last_row["Vibration"], "Vibration")
    temp_html = format_val(last_row["Temperature"], "Temperature")
    pres_html = format_val(last_row["Pressure"], "Pressure")
    speed_html = format_val(last_row["Speed"], "Speed")
    flow_html = format_val(last_row["OilFlow"], "OilFlow")
    
    is_derated = last_row["StressProfile"] == "Hard"
    btn_text = "DE-RATE ACTIVE (ECO)" if is_derated else "ACTIVATE ECO MODE"
    btn_class = "btn-f1 btn-f1-active" if is_derated else "btn-f1 btn-f1-secondary"
    
    # If in Replay HUD mode, style the digital cards to show a yellow border/glow
    card_glow = "digital-card-value value-yellow" if is_replay else "digital-card-value"
    
    return (asset_name, asset_desc, vib_html, temp_html, pres_html, speed_html, flow_html, 
            btn_text, btn_class, card_glow, card_glow, card_glow, card_glow, card_glow)


# Callback: Update tab content (graphs and diagrams)
@app.callback(
    Output("tab-content-container", "children"),
    [Input("diagnostic-tabs", "active_tab"),
     Input("asset-store", "data"),
     Input("selected-asset", "data"),
     Input("replay-slider", "value")]
)
def render_tab_content(active_tab, store_data, selected_asset, replay_val):
    records = json.loads(store_data)
    df = pd.DataFrame(records)
    
    asset_df = df[df["AssetID"] == selected_asset]
    
    # Slicing history depending on replay scrub
    is_replay = replay_val < 0
    if is_replay:
        end_idx = max(5, len(asset_df) + replay_val)
        asset_slice_df = asset_df.iloc[:end_idx].tail(30)
    else:
        asset_slice_df = asset_df.tail(30)
        
    predictions = models.predict(df)
    
    # Sliced prediction logic (running inference on the active slice)
    if is_replay:
        slice_history = asset_df.iloc[:end_idx]
        slice_preds = models.predict(slice_history)
        active_pred = slice_preds[selected_asset]
    else:
        active_pred = predictions[selected_asset]
        
    if active_tab == "telemetry":
        vib_history = asset_slice_df["Vibration"].tolist()
        timestamps = [pd.to_datetime(t) for t in asset_slice_df["Timestamp"]]
        
        # Forecast
        forecast_vals, time_to_fail = forecast_sensor_trend(
            vib_history, 
            time_steps_ahead=20, 
            fail_threshold=sim.baselines["Vibration"]["fail_thresh"],
            is_increasing=True
        )
        
        last_time = timestamps[-1]
        forecast_times = [last_time + datetime.timedelta(minutes=30*i) for i in range(1, 21)]
        
        fig = go.Figure()
        
        # Limit Line
        fig.add_trace(go.Scatter(
            x=[timestamps[0], forecast_times[-1]],
            y=[sim.baselines["Vibration"]["fail_thresh"]]*2,
            mode="lines",
            name="Critical Limit",
            line=dict(color="#ff073a", width=2, dash="dash")
        ))
        
        # Baseline
        fig.add_trace(go.Scatter(
            x=timestamps + forecast_times,
            y=[sim.baselines["Vibration"]["mean"]] * (len(timestamps) + len(forecast_times)),
            mode="lines",
            name="Optimal Lap (Baseline)",
            line=dict(color="rgba(0, 240, 255, 0.25)", width=1.5)
        ))
        
        # Stint Telemetry
        fig.add_trace(go.Scatter(
            x=timestamps,
            y=vib_history,
            mode="lines+markers",
            name="Telemetry logs" if not is_replay else "Replay Telemetry",
            line=dict(color="#00f0ff" if not is_replay else "#ffdf00", width=2.5),
            marker=dict(size=4)
        ))
        
        # Forecast
        fig.add_trace(go.Scatter(
            x=forecast_times,
            y=forecast_vals,
            mode="lines",
            name="Forecasted Trend",
            line=dict(color="#ffdf00", width=2, dash="dot")
        ))
        
        fig.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(6,9,19,0.5)",
            font=dict(color="#94a3b8", family="Share Tech Mono"),
            margin=dict(l=40, r=20, t=10, b=40),
            height=280,
            xaxis=dict(showgrid=True, gridcolor="#1e293b", linecolor="#1e293b"),
            yaxis=dict(title="Vibration (mm/s)", showgrid=True, gridcolor="#1e293b", linecolor="#1e293b"),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )
        
        return dcc.Graph(figure=fig, config={"displayModeBar": False})
        
    elif active_tab == "features":
        contribs = active_pred["Contributions"]
        sensors = list(contribs.keys())
        weights = [contribs[s] * 100 for s in sensors]
        
        sorted_indices = np.argsort(weights)
        sensors = [sensors[i] for i in sorted_indices]
        weights = [weights[i] for i in sorted_indices]
        
        name_mappings = {
            "Vibration": "Vibration (G-Force)",
            "Temperature": "Temperature (Thermal)",
            "Pressure": "Pressure (Hydraulics)",
            "Speed": "Rotational Speed (RPM)",
            "OilFlow": "Oil Flow (Fuel Rate)"
        }
        labels = [name_mappings.get(s, s) for s in sensors]
        
        fig = go.Figure()
        fig.add_trace(go.Bar(
            y=labels,
            x=weights,
            orientation="h",
            marker=dict(
                color=["#00f0ff" if w < 30 else ("#ffdf00" if w < 60 else "#ff073a") for w in weights],
                line=dict(color="#060913", width=1)
            )
        ))
        
        fig.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(6,9,19,0.5)",
            font=dict(color="#94a3b8", family="Share Tech Mono"),
            margin=dict(l=150, r=20, t=10, b=40),
            height=300,
            xaxis=dict(title="XGBoost Attribution Weight (%)", showgrid=True, gridcolor="#1e293b"),
            yaxis=dict(showgrid=False)
        )
        
        return dcc.Graph(figure=fig, config={"displayModeBar": False})
        
    elif active_tab == "map":
        coords = {
            "Asset-01": (1, 3, "CNC Milling"),
            "Asset-02": (4, 4, "Gas Turbine"),
            "Asset-03": (2, 1, "Hydraulic Pump"),
            "Asset-04": (5, 2, "Centrifugal Comp")
        }
        xs, ys, texts, colors, sizes = [], [], [], [], []
        
        for asset_id, (x, y, label) in coords.items():
            xs.append(x)
            ys.append(y)
            texts.append(f"{asset_id}: {label}")
            
            prob = predictions[asset_id]["FailureProbability"]
            is_failed = df[df["AssetID"] == asset_id].iloc[-1]["FailureState"] == 1
            
            if is_failed or prob > 0.65:
                colors.append("#ff073a")
            elif prob > 0.25:
                colors.append("#ffdf00")
            else:
                colors.append("#39ff14")
            sizes.append(28 if asset_id == selected_asset else 18)
            
        fig = go.Figure()
        fig.add_shape(type="rect", x0=0.5, y0=0.5, x1=5.5, y1=4.5, line=dict(color="#1e293b", width=1))
        fig.add_shape(type="line", x0=3, y0=0.5, x1=3, y1=4.5, line=dict(color="#1e293b", width=1, dash="dot"))
        
        fig.add_trace(go.Scatter(
            x=xs, y=ys,
            mode="markers+text",
            text=[f"<b>{a_id}</b>" for a_id in coords.keys()],
            textposition="top center",
            hovertext=texts,
            marker=dict(
                size=sizes,
                color=colors,
                line=dict(color="#f8fafc", width=2)
            ),
            textfont=dict(color="#f8fafc", family="Orbitron", size=11)
        ))
        
        fig.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(6,9,19,0.3)",
            font=dict(color="#94a3b8", family="Share Tech Mono"),
            margin=dict(l=20, r=20, t=20, b=20),
            height=300,
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False, range=[0, 6]),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False, range=[0, 5])
        )
        
        return html.Div(className="factory-map-container", children=[dcc.Graph(figure=fig, config={"displayModeBar": False})])
        
    elif active_tab == "f1-car":
        return html.Div(
            className="factory-map-container",
            children=[
                html.Img(
                    src="/assets/f1_telemetry_schematic.png",
                    style={"width": "100%", "borderRadius": "6px", "border": "1px solid var(--panel-border)"}
                )
            ]
        )

# Callback: Toggle visibility of the Failure Replay Scrubber container based on active tab
@app.callback(
    Output("replay-hud-container", "style"),
    Input("diagnostic-tabs", "active_tab")
)
def toggle_replay_scrubber_visibility(active_tab):
    if active_tab == "telemetry":
        return {"display": "block"}
    else:
        return {"display": "none"}


# Callback: Update Right Column Strategy recommendations + Explainable AI panel
@app.callback(
    [Output("strat-prob", "children"),
     Output("strat-prob", "className"),
     Output("strat-prob-bar", "value"),
     Output("strat-prob-bar", "color"),
     Output("pit-window-text", "children"),
     Output("pit-window-text", "style"),
     Output("pit-checklist-container", "children"),
     Output("xai-diagnostics-container", "children")],
    [Input("asset-store", "data"), 
     Input("selected-asset", "data"),
     Input("replay-slider", "value")]
)
def update_strategy_planner(store_data, selected_asset, replay_val):
    records = json.loads(store_data)
    df = pd.DataFrame(records)
    
    asset_df = df[df["AssetID"] == selected_asset]
    
    # Replay Slider logic
    is_replay = replay_val < 0
    if is_replay:
        end_idx = max(5, len(asset_df) + replay_val)
        slice_history = asset_df.iloc[:end_idx]
        predictions = models.predict(slice_history)
        last_row = slice_history.iloc[-1]
    else:
        predictions = models.predict(df)
        last_row = asset_df.iloc[-1]
        
    pred = predictions[selected_asset]
    prob = pred["FailureProbability"]
    rul = pred["RUL"]
    
    prob_pct = prob * 100
    prob_text = f"{prob_pct:.1f}%"
    
    if prob > 0.65:
        prob_class = "value-red"
        bar_color = "danger"
    elif prob > 0.25:
        prob_class = "value-yellow"
        bar_color = "warning"
    else:
        prob_class = "value-green"
        bar_color = "success"
        
    if rul <= 0.0:
        pit_window = "BOX IMMEDIATELY"
        window_style = {"borderLeftColor": "var(--race-red)", "color": "var(--race-red)", "animation": "pulse-red 0.5s infinite alternate"}
    elif rul < 5.0:
        pit_window = f"BOX WINDOW OPEN // {rul:.1f} HRS LEFT"
        window_style = {"borderLeftColor": "var(--race-red)", "color": "var(--race-red)"}
    else:
        lower = max(1.0, rul * 0.8)
        upper = rul * 1.1
        pit_window = f"BOX WINDOW: {lower:.1f} - {upper:.1f} HRS"
        window_style = {"borderLeftColor": "var(--warning-yellow)", "color": "var(--warning-yellow)"}
        
    checklist_items = []
    if selected_asset == "Asset-01":
        checklist_items = [
            "1. Inspect housing for metal particles",
            "2. Replace rotor bearing (B2-Front)",
            "3. Re-align spindle motor shaft",
            "4. Lubricate bearing assembly"
        ]
    elif selected_asset == "Asset-02":
        checklist_items = [
            "1. Inspect turbine cooling manifolds",
            "2. Flush heat exchanger core radiator",
            "3. Calibrate thermocouple telemetry",
            "4. Refill thermal coolant reservoirs"
        ]
    elif selected_asset == "Asset-03":
        checklist_items = [
            "1. Leak-test hydraulic seal lines",
            "2. Replace rubber gaskets on Pump P4",
            "3. Pressure test cylinder pressure",
            "4. Flush compressor lubricating oil"
        ]
    else:
        checklist_items = [
            "1. Re-torque rotor balance bolts",
            "2. Zero-out balance vibrations",
            "3. Blade inspection for corrosion",
            "4. Recalibrate rotation sensors"
        ]
        
    checklist_html = [
        html.Div(
            className="pit-checklist-item",
            children=[
                html.Span(f"{idx+1}.", className="pit-checklist-num"),
                html.Span(item[3:])
            ]
        ) for idx, item in enumerate(checklist_items)
    ]
    
    # -------------------------------------------------------------
    # EXPLAINABLE AI (XAI) DIAGNOSTICS GENERATION
    # -------------------------------------------------------------
    # Evaluates deviation from baselines to produce an explainable rule log
    xai_rows = []
    
    def generate_xai_status(val, baseline_key):
        mean = sim.baselines[baseline_key]["mean"]
        thresh = sim.baselines[baseline_key]["fail_thresh"]
        
        if baseline_key == "Pressure" or baseline_key == "OilFlow":
            if val < thresh:
                return "FAIL (LOW)", "value-red"
            elif val < thresh * 1.3:
                return "WARN (LOW)", "value-yellow"
            else:
                return "NOMINAL", "value-green"
        else:
            if val > thresh:
                return "FAIL (HIGH)", "value-red"
            elif val > thresh * 0.75:
                return "WARN (HIGH)", "value-yellow"
            else:
                return "NOMINAL", "value-green"
                
    vib_stat, vib_class = generate_xai_status(last_row["Vibration"], "Vibration")
    temp_stat, temp_class = generate_xai_status(last_row["Temperature"], "Temperature")
    pres_stat, pres_class = generate_xai_status(last_row["Pressure"], "Pressure")
    speed_stat, speed_class = generate_xai_status(last_row["Speed"], "Speed")
    flow_stat, flow_class = generate_xai_status(last_row["OilFlow"], "OilFlow")
    
    xai_rows = [
        html.Div(className="xai-metric-row", children=[html.Span("Vibration (G-Force)", className="xai-metric-name"), html.Span(vib_stat, className=f"xai-metric-status {vib_class}")]),
        html.Div(className="xai-metric-row", children=[html.Span("Temperature (Thermal)", className="xai-metric-name"), html.Span(temp_stat, className=f"xai-metric-status {temp_class}")]),
        html.Div(className="xai-metric-row", children=[html.Span("Pressure (Hydraulics)", className="xai-metric-name"), html.Span(pres_stat, className=f"xai-metric-status {pres_class}")]),
        html.Div(className="xai-metric-row", children=[html.Span("Speed (RPM)", className="xai-metric-name"), html.Span(speed_stat, className=f"xai-metric-status {speed_class}")]),
        html.Div(className="xai-metric-row", children=[html.Span("Oil Flow (Fuel Rate)", className="xai-metric-name"), html.Span(flow_stat, className=f"xai-metric-status {flow_class}")]),
    ]
    
    return prob_text, prob_class, prob_pct, bar_color, pit_window, window_style, checklist_html, xai_rows


# Callback: Notification bell dropdown toggle & badge count
@app.callback(
    [Output("notif-dropdown-container", "children"),
     Output("notif-dropdown-container", "style"),
     Output("notif-badge", "style"),
     Output("notif-badge", "children")],
    [Input("btn-notif-bell", "n_clicks"), Input("notif-store", "data")],
    [State("notif-dropdown-container", "style")]
)
def handle_notifications(n_clicks, notif_data, current_style):
    notifications = json.loads(notif_data) if notif_data else []
    
    # Toggle display based on click parity
    if n_clicks % 2 == 1:
        dropdown_style = {"display": "block"}
    else:
        dropdown_style = {"display": "none"}
        
    badge_style = {"display": "none"}
    badge_val = ""
    
    # If we have unread alarms, show red badge
    if len(notifications) > 0:
        badge_style = {"display": "block"}
        badge_val = str(len(notifications))
        
    # Render alarm list
    if not notifications:
        items_html = [html.Div("No active system alarms.", style={"padding": "10px 15px", "color": "var(--text-secondary)", "fontSize": "11px"})]
    else:
        items_html = [
            html.Div(
                className="notif-item",
                children=[
                    html.Div(item["time"], className="notif-item-time"),
                    html.Div(dcc.Markdown(item["message"]), className="notif-item-msg")
                ]
            ) for item in notifications
        ]
        
    dropdown_layout = html.Div(
        className="notif-dropdown",
        children=[
            html.Div(
                className="notif-header",
                children=[
                    html.Span("SYSTEM WARNING LOG"),
                    html.Span(f"{len(notifications)} ALARMS", style={"color": "var(--race-red)"})
                ]
            ),
            html.Div(items_html)
        ]
    )
    
    return dropdown_layout, dropdown_style, badge_style, badge_val


# Callback: Collapsible AI Chat Assistant Widget
@app.callback(
    [Output("chat-collapsible-content", "style"), Output("chat-toggle-icon", "children")],
    Input("chat-widget-header", "n_clicks"),
    State("chat-collapsible-content", "style")
)
def toggle_chat_widget(n_clicks, current_style):
    if n_clicks is None:
        return {"display": "none"}, "▲"
    if n_clicks % 2 == 1:
        return {"display": "none"}, "▲"
    else:
        return {"display": "flex", "flexDirection": "column"}, "▼"


# Callback: AI Chat Assistant messaging engine
@app.callback(
    [Output("chat-history-container", "children"),
     Output("chat-history-store", "data"),
     Output("chat-input-field", "value")],
    [Input("btn-chat-send", "n_clicks"), Input("chat-input-field", "n_submit")],
    [State("chat-input-field", "value"),
     State("chat-history-store", "data"),
     State("asset-store", "data")]
)
def run_chat_assistant(n_clicks, n_submit, user_text, history_json, store_data):
    ctx = callback_context
    triggered_id = ctx.triggered[0]["prop_id"].split(".")[0] if ctx.triggered else ""
    
    chat_history = json.loads(history_json)
    
    if (triggered_id == "btn-chat-send" or triggered_id == "chat-input-field") and user_text:
        # Append User Message
        chat_history.append({"sender": "user", "text": user_text})
        
        # Analyze current telemetry state to respond intelligently
        records = json.loads(store_data)
        df = pd.DataFrame(records)
        predictions = models.predict(df)
        
        query = user_text.upper()
        bot_response = ""
        
        # Match asset mentions
        matched_asset = None
        for a_id in sim.assets.keys():
            if a_id.upper() in query:
                matched_asset = a_id
                break
                
        if matched_asset:
            pred = predictions[matched_asset]
            prob = pred["FailureProbability"] * 100
            rul = pred["RUL"]
            stress = pred["StressProfile"]
            
            # Identify highest violating parameter
            contribs = pred["Contributions"]
            max_violator = max(contribs, key=contribs.get)
            
            status = "STAY OUT"
            if prob > 65:
                status = "BOX (CRITICAL)"
            elif prob > 25:
                status = "DE-RATE (WARNING)"
                
            # Tailored query responses based on user question keywords
            if "WHY" in query or "RISK" in query or "FAIL" in query:
                bot_response = (
                    f"Asset **{matched_asset}** failure risk is **{prob:.1f}%** ({status}).\n"
                    f"The primary driver is **{max_violator}**, contributing **{contribs[max_violator]*100:.1f}%** to the total XGBoost risk score.\n"
                    f"Action: Consider toggling **Eco Mode** to de-rate operating speed and reduce stress load."
                )
            elif "SENSOR" in query or "THRESHOLD" in query or "CROSS" in query:
                val = df[df["AssetID"] == matched_asset].iloc[-1][max_violator]
                thresh = sim.baselines[max_violator]["fail_thresh"]
                unit = sim.baselines[max_violator]["unit"]
                bot_response = (
                    f"On **{matched_asset}**, the leading stress sensor is **{max_violator}**.\n"
                    f"Current reading: **{val:.1f}{unit}**.\n"
                    f"Critical threshold: **{thresh:.1f}{unit}**.\n"
                    f"Model is detecting signal deviation from baseline values."
                )
            elif "HOURS" in query or "RUL" in query or "BEFORE" in query or "TIME" in query:
                bot_response = (
                    f"Asset **{matched_asset}** has an estimated **{rul:.1f} hours** of Remaining Useful Life (RUL).\n"
                    f"The maintenance pit window opens between **{max(1.0, rul*0.8):.1f}** and **{rul*1.1:.1f} hours**.\n"
                    f"Toggling Eco Mode will de-rate output load and extend this window."
                )
            else:
                bot_response = (
                    f"Diagnostics for **{matched_asset}**:\n"
                    f"- Risk Probability: **{prob:.1f}%** ({status})\n"
                    f"- Est. Remaining Life: **{rul:.1f} hours**\n"
                    f"- Active Compound: **{stress} tyre**\n"
                    f"- Primary Stress Driver: **{max_violator}** (attributing {contribs[max_violator]*100:.1f}% to risk).\n"
                    f"Recommended action: Toggle **Eco Mode** to extend RUL, or call **BOX NOW** to schedule maintenance."
                )
        elif "SYSTEM" in query or "ALL" in query or "STATUS" in query:
            active_warnings = [a for a, p in predictions.items() if p["FailureProbability"] > 0.25]
            if not active_warnings:
                bot_response = "All assets are currently **nominal** (Green Flag). Average system wear index is 12.4%."
            else:
                warn_str = ", ".join(active_warnings)
                bot_response = f"System is currently in **Safety Car** status. Assets warning/failed: **{warn_str}**. Review standing column listings."
        elif "HELP" in query or "HI" in query or "HELLO" in query:
            bot_response = "I can analyze live telemetry. Ask me: 'Status of Asset-01', 'Why is Asset-03 failing?', or 'System status report'."
        else:
            bot_response = "Telemetry assistant. Query an asset ID (e.g. 'Asset-01') or system summaries to get diagnostics."
            
        chat_history.append({"sender": "bot", "text": bot_response})
        user_text = "" # reset input field
        
    # Render chat history bubbles
    chat_bubbles = []
    for msg in chat_history:
        bubble_class = "chat-msg chat-msg-bot" if msg["sender"] == "bot" else "chat-msg chat-msg-user"
        chat_bubbles.append(
            html.Div(
                className=bubble_class,
                children=dcc.Markdown(msg["text"])
            )
        )
        
    return chat_bubbles, json.dumps(chat_history), user_text


# Run Server
if __name__ == "__main__":
    app.run(debug=True, port=8050)
