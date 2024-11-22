import dash
from dash import dcc, html, Input, Output, State
import pandas as pd
import xgboost as xgb

import plotly.express as px

# Load your data and model
X_test = pd.read_csv('2_data/X_test_dashboard.csv')
y_test = pd.read_csv('2_data/y_test_dashboard.csv')
data = pd.merge(X_test, y_test, on='Customer ID')

model = xgb.Booster()
import os

model_path = './6_dashboard/xgboost_model.json'
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
        customer_data = data[data['Customer ID'] == customer_id]
        if customer_data.empty:
            return "Customer ID not found", "", ""
        
        # Predict churn
        dmatrix = xgb.DMatrix(customer_data.drop(columns=['Customer ID', 'Churn']))
        churn_prob = model.predict(dmatrix)[0]
        churn_prediction = "Likely to Churn" if churn_prob > 0.5 else "Not Likely to Churn"
        
        # Feature differences
        #eature_diff = pd.Series()
        feature_diff = pd.Series()
        feature_diff_fig = None
        if churn_prob > 0.5:
            non_churn_data = data[data['Churn'] == 0].drop(columns=['Customer ID']).mean()
            feature_diff = customer_data.drop(columns=['Customer ID', 'Churn']).iloc[0] - non_churn_data
            feature_diff = feature_diff[feature_diff > 0].sort_values(ascending=False).head(7)
            feature_diff_fig = px.bar(feature_diff, title="Top 7 Feature Differences from Non-Churn Customers")
            feature_diff_fig.update_layout(yaxis_title="Feature", xaxis_title="Difference")

        # Define actionable features and their corresponding recommendations
        actionable_features = {
            'Satisfaction Score': "Improve customer satisfaction by addressing their concerns and providing better service.",
            'Referred a Friend': "Encourage the customer to refer friends by offering incentives or rewards.",
            'Number of Referrals': "Highlight referral programs and offer additional rewards for new referrals.",
            'Online Security': "Highlight the benefits of your online security features to make the customer feel safer.",
            'Online Backup': "Promote the importance of online backup services and offer special discounts or upgrades.",
            'Device Protection Plan': "Explain the advantages of the device protection plan and offer a trial or discount.",
            'Premium Tech Support': "Emphasize the value of premium tech support, including faster response times and better support.",
            'Streaming TV': "Highlight streaming TV features and offer promotions for enhanced services.",
            'Streaming Movies': "Encourage the use of streaming movie services by showcasing benefits or offering package deals.",
            'Streaming Music': "Promote streaming music services by mentioning exclusive content or additional benefits.",
            'Unlimited Data': "Promote the benefits of unlimited data plans to customers concerned about data limits.",
            'Paperless Billing': "Encourage customers to opt for paperless billing by emphasizing convenience and environmental benefits.",
            'Contract_Month-to-Month': "Promote the benefits of long-term contracts, such as discounts or added features, to month-to-month customers.",
            'Contract_One Year': "Encourage one-year contract customers to upgrade to longer-term plans by highlighting cost savings.",
            'Contract_Two Year': "Reward long-term customers by discussing perks or loyalty benefits.",
            'Payment Method_Bank Withdrawal': "Encourage customers to use bank withdrawal for payment by explaining convenience and reliability.",
            'Payment Method_Credit Card': "Recommend credit card payments for easier tracking and potential reward points.",
            'Payment Method_Mailed Check': "Suggest alternative payment methods like online banking or auto-pay for more convenience."
        }

        # Generate recommendations based on feature differences
        recommendations = [
            recommendation for feature, recommendation in actionable_features.items()
            if feature_diff.get(feature, 0) > 0
        ]

        # Non-actionable features
        non_actionable_features = [
            'Gender', 'Age', 'Senior Citizen', 'Married', 'Dependents', 
            'Number of Dependents', 'City', 'Zip Code', 'Latitude', 
            'Longitude', 'Population', 'Country', 'State', 'Quarter', 
            'Avg Monthly Long Distance Charges', 'Total Charges', 
            'Total Refunds', 'Total Extra Data Charges', 'Total Long Distance Charges',
            'Total Revenue', 'CLTV', 'Tenure', 'Monthly Charges', 
            'Monthly_Charges_Scaled', 'Refund_to_Charges_Ratio', 
            'Extra_Data_Usage_Cost_Proportion', 'Lifetime_Value_per_Month'
        ]

        non_actionable_diff_features = [feature for feature in feature_diff.index if feature in non_actionable_features]
        if non_actionable_diff_features:
            recommendations.append(
            f"The following features are not actionable during customer service calls: {', '.join(non_actionable_diff_features)}."
            )

        return (
            f"Customer ID: {customer_id}",
            f"Churn Prediction: {churn_prediction}",
            html.Div([
            dcc.Graph(figure=feature_diff_fig),
            html.H3("Recommendations to Retain Customer:"),
            html.Ul([html.Li(rec) for rec in recommendations])
            ])
        )
        
        return (
            f"Customer ID: {customer_id}",
            f"Churn Prediction: {churn_prediction}",
            dcc.Graph(figure=feature_diff_fig)
        )
    return "", "", ""

if __name__ == '__main__':
    app.run_server(debug=True)