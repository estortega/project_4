from dash import dcc, html
from dash.dependencies import Input, Output
import plotly.express as px
import sqlite3
import pandas as pd
import numpy as np
import joblib

# Load your trained model
model = joblib.load("final_model.pkl")

# Start the Dash app
import dash
app = dash.Dash(__name__)
app.title = "Credit Card Fraud Dashboard"

app.layout = html.Div([
    html.H2("Live Fraud Prediction"),
    html.Button("Get Random Transaction", id="btn", n_clicks=0),
    html.Div(id="prediction-output"),
    dcc.Graph(id="feature-graph")
])

@app.callback(
    [Output("prediction-output", "children"),
     Output("feature-graph", "figure")],
    [Input("btn", "n_clicks")]
)
def update_output(n_clicks):
    # Read 1 random transaction from the database
    conn = sqlite3.connect("creditcard.db")
    tx = pd.read_sql("SELECT * FROM transactions ORDER BY RANDOM() LIMIT 1", conn)
    conn.close()

    tx_id = tx.index[0]
    features = tx.drop(columns=["Class"])
    prediction = model.predict(features)[0]
    label = "FRAUD ❌" if prediction == 1 else "Not Fraud ✅"

    # Highlight top 3 features by absolute magnitude
    values = features.values[0]
    columns = features.columns
    top_indices = np.argsort(np.abs(values))[-3:]
    colors = ['red' if i in top_indices else 'lightgray' for i in range(len(values))]

    fig = px.bar(x=columns, y=values, labels={"x": "Feature", "y": "Value"})
    fig.update_traces(marker_color=colors)
    fig.update_layout(title=f"Prediction for Transaction #{tx_id}: {label}")

    return f"Prediction for Transaction #{tx_id}: {label}", fig

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=1000, debug=True)

