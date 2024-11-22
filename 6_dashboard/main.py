import dash
from dash import dcc, html, Input, Output, State
import pandas as pd
import xgboost as xgb

import plotly.express as px

# Load your data and model
data = pd.read_csv('2_data/X_test.csv')
model = xgb.Booster()
import os

model_path = 'xgboost_model.json'
if not os.path.exists(model_path):
    raise FileNotFoundError(f"Model file not found: {model_path}")
model.load_model(model_path)

# Initialize the Dash app
app = dash.Dash(__name__)

app.layout = html.Div([
    html.H1("Telco Customer Churn Dashboard"),
    dcc.Input(id='customer-id-input', type='text', placeholder='Enter Customer ID'),
    html.Button('Search', id='search-button', n_clicks=0),
    html.Div(id='customer-info'),
    html.Div(id='churn-prediction'),
    html.Div(id='feature-differences')
])

@app.callback(
    [Output('customer-info', 'children'),
     Output('churn-prediction', 'children'),
     Output('feature-differences', 'children')],
    [Input('search-button', 'n_clicks')],
    [State('customer-id-input', 'value')]
)
def update_dashboard(n_clicks, customer_id):
    if n_clicks > 0 and customer_id:
        customer_data = data[data['customerID'] == customer_id]
        if customer_data.empty:
            return "Customer ID not found", "", ""
        
        # Predict churn
        dmatrix = xgb.DMatrix(customer_data.drop(columns=['customerID', 'Churn']))
        churn_prob = model.predict(dmatrix)[0]
        churn_prediction = "Likely to Churn" if churn_prob > 0.5 else "Not Likely to Churn"
        
        # Feature differences
        non_churn_data = data[data['Churn'] == 0].mean()
        feature_diff = customer_data.drop(columns=['customerID', 'Churn']).iloc[0] - non_churn_data
        feature_diff = feature_diff.sort_values(ascending=False)
        feature_diff_fig = px.bar(feature_diff, title="Feature Differences from Non-Churn Customers")
        
        return (
            f"Customer ID: {customer_id}",
            f"Churn Prediction: {churn_prediction}",
            dcc.Graph(figure=feature_diff_fig)
        )
    return "", "", ""

if __name__ == '__main__':
    app.run_server(debug=True)