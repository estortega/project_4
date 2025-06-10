
import joblib
import pandas as pd
from dash import Dash, html, dcc, Input, Output
import plotly.express as px
from flask import Flask

# Load model
model = joblib.load("model/final_model.pkl")

# Load sample data
df = pd.read_csv("data/creditcard_sample.csv")

# Create Flask server and wrap with Dash
server = Flask(__name__)
app = Dash(__name__, server=server)

app.layout = html.Div([
    html.H1("Credit Card Fraud Detection Dashboard"),
    dcc.Dropdown(
        id='sample-input',
        options=[{'label': f"Transaction {i}", 'value': i} for i in df.index[:10]],
        placeholder="Select a transaction to evaluate",
    ),
    html.Div(id='model-output', style={"marginTop": 20}),
    html.H3("Transaction Class Distribution"),
    dcc.Graph(figure=px.histogram(df, x="Class", title="Fraud vs Non-Fraud"))
])

@app.callback(
    Output('model-output', 'children'),
    Input('sample-input', 'value')
)
def predict_fraud(sample_index):
    if sample_index is None:
        return "Select a transaction to make a prediction."
    row = df.drop("Class", axis=1).iloc[sample_index].values.reshape(1, -1)
    prediction = model.predict(row)[0]
    return f"Prediction: {'Fraudulent' if prediction == 1 else 'Legit'}"

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=8050)
