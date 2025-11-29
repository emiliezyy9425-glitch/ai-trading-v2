import logging
import sys
import pandas as pd
import plotly.graph_objs as go
from dash import Dash, dcc, html, Input, Output
import os
import json

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler("logs/dashboard.log"),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

app = Dash(__name__)
app.title = "TSLA AI Trade Dashboard"

EXPECTED_COLS = ["timestamp","ticker","price","decision","fib","tds","td9","RSI","MACD","Signal","Volume","IV","Delta","Source"]

def _find_first_existing(paths):
    for p in paths:
        if os.path.exists(p) and os.path.getsize(p) > 0:
            return p
    return None

def load_trade_log():
    # Support both live and generator outputs
    candidate_paths = ["data/trade_log.csv", "logs/trade_log.csv"]
    path = _find_first_existing(candidate_paths)
    if not path:
        logger.warning("⚠️ Trade log not found at %s", candidate_paths)
        return pd.DataFrame(columns=EXPECTED_COLS)
    try:
        # Try headered CSV first (generator), else fallback to header=None (old/plain formats)
        df = pd.read_csv(path)
        if not set(EXPECTED_COLS).issubset(df.columns):
            df = pd.read_csv(path, header=None, names=EXPECTED_COLS)
        # Try to coerce types
        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
        for c in ["price","RSI","MACD","Signal","Volume","IV","Delta"]:
            if c in df.columns:
                df[c] = pd.to_numeric(df[c], errors="coerce")
        return df.dropna(subset=["timestamp"])
    except Exception as e:
        logger.error("❌ Error loading trade log: %s", e)
        return pd.DataFrame(columns=EXPECTED_COLS)

def load_metrics():
    # Prefer backtester JSON, fallback to CSV if you add one later
    json_path = "data/backtest_metrics.json"
    csv_path = "logs/backtest_metrics.csv"
    try:
        if os.path.exists(json_path):
            with open(json_path, "r") as f:
                m = json.load(f)
            return pd.DataFrame([m])
        if os.path.exists(csv_path):
            return pd.read_csv(csv_path)
        logger.warning("⚠️ No metrics found at %s / %s", json_path, csv_path)
        return pd.DataFrame()
    except Exception as e:
        logger.error("❌ Error loading metrics: %s", e)
        return pd.DataFrame()

@app.callback(
    [Output("trade-graph", "figure"), Output("metrics-table", "children")],
    [Input("refresh-btn", "n_clicks")]
)
def update_graph_and_metrics(n_clicks):
    df = load_trade_log()
    if df.empty:
        fig = go.Figure(data=[], layout={
            "title": "TSLA AI Trade Log",
            "xaxis": {"title": "Time"},
            "yaxis": {"title": "Price"},
            "yaxis2": {"title": "Indicators", "overlaying": "y", "side": "right", "showgrid": False},
            "showlegend": True
        })
    else:
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=df["timestamp"], y=df["price"], mode="lines+markers", name="TSLA Price"))
        if "RSI" in df.columns:
            fig.add_trace(go.Scatter(x=df["timestamp"], y=df["RSI"], mode="lines", name="RSI", yaxis="y2"))
        if "MACD" in df.columns:
            fig.add_trace(go.Scatter(x=df["timestamp"], y=df["MACD"], mode="lines", name="MACD", yaxis="y2"))

        buy = df[df["decision"].astype(str).str.lower().str.contains("buy", na=False)]
        fig.add_trace(go.Scatter(x=buy["timestamp"], y=buy["price"],
                                 mode="markers", marker=dict(color="green", size=10), name="Buy Signals"))

        sell = df[df["decision"].astype(str).str.lower().str.contains("sell", na=False)]
        fig.add_trace(go.Scatter(x=sell["timestamp"], y=sell["price"],
                                 mode="markers", marker=dict(color="red", size=10), name="Sell Signals"))

        fig.update_layout(
            title="TSLA AI Trade Log",
            xaxis_title="Time",
            yaxis=dict(title="Price", side="left"),
            yaxis2=dict(title="Indicators", overlaying='y', side='right', showgrid=False),
            legend_title_text="Signals",
            hovermode="x unified"
        )

    metrics_df = load_metrics()
    metrics_table = (
        html.P("No metrics available") if metrics_df.empty else
        html.Table([
            html.Tr([html.Th(col) for col in metrics_df.columns])] + [
            html.Tr([html.Td(metrics_df.iloc[i][col]) for col in metrics_df.columns]) for i in range(len(metrics_df))
        ])
    )

    return fig, metrics_table

app.layout = html.Div([
    html.H1("📈 TSLA AI Trade Dashboard"),
    html.Button("🔄 Refresh", id="refresh-btn", n_clicks=0),
    dcc.Graph(id="trade-graph"),
    html.Div(id="metrics-table")
])

if __name__ == "__main__":
    logger.info("🚀 Starting dashboard at http://%s:%d", "0.0.0.0", 8050)
    app.run(host="0.0.0.0", port=8050)
