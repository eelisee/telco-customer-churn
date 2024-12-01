import dash
from dash import dcc, html, Input, Output, State
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import os
import dash_bootstrap_components as dbc
import joblib
import pickle

# Load your data and model
X_test = pd.read_csv('2_data/X_test_dashboard.csv')
y_test = pd.read_csv('2_data/y_test_dashboard.csv')
data = pd.merge(X_test, y_test, on='Customer ID')

# Load the Decision Tree model
model_path = './6_dashboard/decisiontree_model.pkl'
with open(model_path, 'rb') as f:
    model = pickle.load(f)

# Initialize the Dash app
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.FLATLY])
app.title = "Telco Customer Churn Dashboard"

app.layout = html.Div([
    html.Div([
        html.H1("Telco Customer Churn Dashboard", style={'textAlign': 'center'}),
        dcc.Input(id='customer-id-input', type='text', placeholder='Enter Customer ID', style={'width': '50%'}),
        html.Button('Search', id='search-button', n_clicks=0, style={'marginLeft': '10px'}),
    ], style={'textAlign': 'center', 'padding': '20px'}),
    
    html.Div(id='customer-info', style={'padding': '20px'}),
    html.Div(id='churn-prediction', style={'padding': '20px'}),
    html.Div(id='feature-differences', style={'padding': '20px'}),
    html.Div(id='recommendations', style={'padding': '20px'})
])

@app.callback(
    [Output('customer-info', 'children'),
     Output('churn-prediction', 'children'),
     Output('feature-differences', 'children'),
     Output('recommendations', 'children')],
    [Input('search-button', 'n_clicks')],
    [State('customer-id-input', 'value')]
)
def update_dashboard(n_clicks, customer_id):
    if n_clicks > 0 and customer_id:
        customer_data = data[data['Customer ID'] == customer_id]
        if customer_data.empty:
            return "Customer ID not found", "", "", ""
        
        # Predict churn
        customer_features = customer_data.drop(columns=['Customer ID', 'Churn']).values
        churn_prob = model.predict_proba(customer_features)[0][1]
        churn_prediction = "Likely to Churn" if churn_prob > 0.5 else "Not Likely to Churn"
        
        # Feature differences
        feature_diff = pd.Series()
        feature_diff_fig = None
        if churn_prob > 0.5:
            non_churn_data = data[data['Churn'] == 0].drop(columns=['Customer ID']).mean()
            feature_diff = customer_data.drop(columns=['Customer ID', 'Churn']).iloc[0] - non_churn_data
            feature_diff = feature_diff.sort_values(ascending=False).head(7)
            feature_diff_fig = go.Figure()
            feature_diff_fig.add_trace(go.Scatter(
                x=feature_diff.index,
                y=feature_diff.values,
                mode='markers',
                marker=dict(size=12, color='red'),
                name='Customer'
            ))
            feature_diff_fig.add_trace(go.Scatter(
                x=feature_diff.index,
                y=[0]*len(feature_diff),
                mode='lines',
                name='Non-Churn Median'
            ))
            feature_diff_fig.update_layout(
                title="Top 7 Feature Differences from Non-Churn Customers",
                yaxis_title="Difference",
                xaxis_title="Feature",
                yaxis=dict(tickformat=".2f")
            )

        # Define actionable features and their corresponding recommendations
        actionable_features = {
            'Satisfaction Score': "Improve customer satisfaction.",
            'Referred a Friend': "Encourage referrals.",
            'Number of Referrals': "Highlight referral programs.",
            'Online Security': "Promote online security features.",
            'Online Backup': "Offer discounts on online backup.",
            'Device Protection Plan': "Explain device protection benefits.",
            'Premium Tech Support': "Emphasize premium tech support.",
            'Streaming TV': "Promote streaming TV features.",
            'Streaming Movies': "Showcase streaming movie benefits.",
            'Streaming Music': "Promote streaming music services.",
            'Unlimited Data': "Highlight unlimited data plans.",
            'Paperless Billing': "Encourage paperless billing.",
            'Contract_Month-to-Month': "Promote long-term contracts.",
            'Contract_One Year': "Encourage longer-term plans.",
            'Contract_Two Year': "Discuss loyalty benefits.",
            'Payment Method_Bank Withdrawal': "Explain bank withdrawal convenience.",
            'Payment Method_Credit Card': "Recommend credit card payments.",
            'Payment Method_Mailed Check': "Suggest online banking or auto-pay."
        }

        # Generate recommendations based on feature differences
        recommendations = [
            recommendation for feature, recommendation in actionable_features.items()
            if feature_diff.get(feature, 0) > 0
        ]

        # General recommendations based on feature importance
        general_recommendations = []
        feature_importance = {
            'Satisfaction Score': 0.573,
            'Online Security': 0.0606,
            'Contract_Month-to-Month': 0.0483,
            'Contract_Two Year': 0.0157,
            'Dependents': 0.0241,
            'Referred a Friend': 0.0198,
            'Monthly_Charges_Scaled': 0.0199,
            'Tech Support_No': 0.0199,
            'Offer E': 0.0094,
            'Streaming Services Count': 0.0088,
            'Payment Method_Credit Card': 0.0087,
            'Age': 0.0064,
            'Lifetime_Value_per_Month': 0.0060
        }

        for feature, importance in feature_importance.items():
            if feature in customer_data.columns and feature not in feature_diff.index:
                value = customer_data[feature].values[0]
                if feature == 'Satisfaction Score' and value < 7:
                    general_recommendations.append("Low satisfaction. Offer personalized support.")
                elif feature == 'Online Security' and value == 0:
                    general_recommendations.append("No online security. Highlight benefits.")
                elif feature == 'Contract_Month-to-Month' and value == 1:
                    general_recommendations.append("Month-to-month contract. Promote long-term benefits.")
                elif feature == 'Contract_Two Year' and value == 0:
                    general_recommendations.append("Not on two-year contract. Discuss perks.")
                elif feature == 'Dependents' and value == 0:
                    general_recommendations.append("No dependents. Highlight family plans.")
                elif feature == 'Referred a Friend' and value == 0:
                    general_recommendations.append("No referrals. Encourage referrals.")
                elif feature == 'Monthly_Charges_Scaled' and value > 0.8:
                    general_recommendations.append("High monthly charges. Discuss cost-saving plans.")
                elif feature == 'Tech Support_No' and value == 1:
                    general_recommendations.append("No tech support. Emphasize premium support.")
                elif feature == 'Offer E' and value == 0:
                    general_recommendations.append("No Offer E. Highlight benefits.")
                elif feature == 'Streaming Services Count' and value < 3:
                    general_recommendations.append("Few streaming services. Promote bundles.")
                elif feature == 'Payment Method_Credit Card' and value == 0:
                    general_recommendations.append("No credit card payment. Recommend for convenience.")
                elif feature == 'Age' and value < 30:
                    general_recommendations.append("Young customer. Highlight appealing services.")
                elif feature == 'Lifetime_Value_per_Month' and value < 50:
                    general_recommendations.append("Low lifetime value. Discuss engagement strategies.")

        combined_recommendations = recommendations + general_recommendations

        # Cluster recommendations into categories
        clustered_recommendations = {
            "Customer Satisfaction": [rec for rec in combined_recommendations if "satisfaction" in rec.lower()],
            "Security and Support": [rec for rec in combined_recommendations if "security" in rec.lower() or "support" in rec.lower()],
            "Contracts and Payments": [rec for rec in combined_recommendations if "contract" in rec.lower() or "payment" in rec.lower()],
            "Streaming and Services": [rec for rec in combined_recommendations if "streaming" in rec.lower() or "services" in rec.lower()],
            "Referrals and Offers": [rec for rec in combined_recommendations if "referral" in rec.lower() or "offer" in rec.lower()]
        }

        return (
            f"Customer ID: {customer_id}",
            f"Churn Prediction: {churn_prediction}",
            html.Div([
                dcc.Graph(figure=feature_diff_fig),
                html.H3("Recommendations to Retain Customer:"),
                html.Div([
                    html.H4("Customer Satisfaction"),
                    html.Ul([html.Li(rec) for rec in clustered_recommendations["Customer Satisfaction"]]),
                    html.H4("Security and Support"),
                    html.Ul([html.Li(rec) for rec in clustered_recommendations["Security and Support"]]),
                    html.H4("Contracts and Payments"),
                    html.Ul([html.Li(rec) for rec in clustered_recommendations["Contracts and Payments"]]),
                    html.H4("Streaming and Services"),
                    html.Ul([html.Li(rec) for rec in clustered_recommendations["Streaming and Services"]]),
                    html.H4("Referrals and Offers"),
                    html.Ul([html.Li(rec) for rec in clustered_recommendations["Referrals and Offers"]])
                ])
            ]),
            ""
        )
    return "", "", "", ""

if __name__ == '__main__':
    app.run_server(debug=True)
