import streamlit as st
import pandas as pd
import pickle

st.set_page_config(
    page_title="Credit Card Fraud Detection",
    page_icon="💳",
    layout="wide"
)

# Load model
try:
    with open("ensemble_fraud_model.pkl", "rb") as file:
        ensemble_model = pickle.load(file)

    with open("feature_columns.pkl", "rb") as file:
        feature_columns = pickle.load(file)

except Exception as e:
    st.error("Error loading model files")
    st.write(e)
    st.stop()

# Title
st.title("💳 Credit Card Fraud Detection")
st.subheader("Ensemble Classification Model")

st.write("Enter transaction details and click the Predict button.")

st.divider()

# Input columns
col1, col2, col3 = st.columns(3)

with col1:
    amount_usd = st.number_input(
        "Transaction Amount (USD)",
        min_value=0.0,
        value=100.0
    )

    merchant_category = st.selectbox(
        "Merchant Category",
        ["grocery", "restaurant", "travel", "electronics",
         "clothing", "entertainment", "online"]
    )

    card_type = st.selectbox(
        "Card Type",
        ["credit", "debit"]
    )

    auth_method = st.selectbox(
        "Authentication Method",
        ["pin", "otp", "biometric", "password"]
    )

    channel = st.selectbox(
        "Transaction Channel",
        ["online", "pos", "mobile", "atm"]
    )

    device_type = st.selectbox(
        "Device Type",
        ["mobile", "desktop", "tablet"]
    )

    is_foreign_transaction = st.selectbox(
        "Foreign Transaction",
        [0, 1]
    )

    hours_since_last_txn = st.number_input(
        "Hours Since Last Transaction",
        min_value=0.0,
        value=5.0
    )

with col2:
    txn_count_last_24h = st.number_input(
        "Transaction Count Last 24 Hours",
        min_value=0,
        value=2
    )

    distance_from_home_km = st.number_input(
        "Distance From Home (km)",
        min_value=0.0,
        value=10.0
    )

    card_age_months = st.number_input(
        "Card Age (Months)",
        min_value=0,
        value=24
    )

    customer_age = st.number_input(
        "Customer Age",
        min_value=18,
        value=30
    )

    account_balance_usd = st.number_input(
        "Account Balance (USD)",
        min_value=0.0,
        value=5000.0
    )

    is_new_merchant = st.selectbox(
        "New Merchant",
        [0, 1]
    )

    used_vpn = st.selectbox(
        "VPN Used",
        [0, 1]
    )

    ip_country_mismatch = st.selectbox(
        "IP Country Mismatch",
        [0, 1]
    )

with col3:
    billing_shipping_mismatch = st.selectbox(
        "Billing Shipping Mismatch",
        [0, 1]
    )

    cvv_retry_count = st.number_input(
        "CVV Retry Count",
        min_value=0,
        value=0
    )

    velocity_score = st.number_input(
        "Velocity Score",
        min_value=0.0,
        value=20.0
    )

    time_of_day_hour = st.number_input(
        "Time Of Day Hour",
        min_value=0,
        max_value=23,
        value=12
    )

    day_of_week = st.number_input(
        "Day Of Week",
        min_value=0,
        max_value=6,
        value=1
    )

    is_ai_generated_scam_attempt = st.selectbox(
        "AI Generated Scam Attempt",
        [0, 1]
    )

    merchant_risk_score = st.number_input(
        "Merchant Risk Score",
        min_value=0.0,
        value=20.0
    )

    prior_disputes = st.number_input(
        "Prior Disputes",
        min_value=0,
        value=0
    )

st.divider()

# Prediction button
if st.button("🔍 Predict Transaction", type="primary"):

    input_data = pd.DataFrame({
        "amount_usd": [amount_usd],
        "merchant_category": [merchant_category],
        "card_type": [card_type],
        "auth_method": [auth_method],
        "channel": [channel],
        "device_type": [device_type],
        "is_foreign_transaction": [is_foreign_transaction],
        "hours_since_last_txn": [hours_since_last_txn],
        "txn_count_last_24h": [txn_count_last_24h],
        "distance_from_home_km": [distance_from_home_km],
        "card_age_months": [card_age_months],
        "customer_age": [customer_age],
        "account_balance_usd": [account_balance_usd],
        "is_new_merchant": [is_new_merchant],
        "used_vpn": [used_vpn],
        "ip_country_mismatch": [ip_country_mismatch],
        "billing_shipping_mismatch": [billing_shipping_mismatch],
        "cvv_retry_count": [cvv_retry_count],
        "velocity_score": [velocity_score],
        "time_of_day_hour": [time_of_day_hour],
        "day_of_week": [day_of_week],
        "is_ai_generated_scam_attempt": [is_ai_generated_scam_attempt],
        "merchant_risk_score": [merchant_risk_score],
        "prior_disputes": [prior_disputes]
    })

    # Encode categorical columns
    input_data = pd.get_dummies(
        input_data,
        columns=[
            "merchant_category",
            "card_type",
            "auth_method",
            "channel",
            "device_type"
        ],
        drop_first=True
    )

    # Match training columns
    input_data = input_data.reindex(
        columns=feature_columns,
        fill_value=0
    )

    input_data = input_data.astype(int)

    # Prediction
    prediction = ensemble_model.predict(input_data)[0]

    if hasattr(ensemble_model, "predict_proba"):
        probability = ensemble_model.predict_proba(input_data)[0][1]
    else:
        probability = 0

    if prediction == 1:
        st.error("🚨 FRAUD TRANSACTION DETECTED")
    else:
        st.success("✅ NORMAL TRANSACTION")

    st.write(
        f"Fraud Probability: **{probability * 100:.2f}%**"
    )

st.divider()

st.caption(
    "Ensemble Model: Random Forest + Decision Tree + Gradient Boosting"
)