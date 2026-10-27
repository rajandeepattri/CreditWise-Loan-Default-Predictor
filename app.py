import streamlit as st
import pandas as pd
import joblib

st.set_page_config(
    page_title="CreditWise",
    page_icon="📊",
    layout="wide"
)

@st.cache_resource
def load_model():
    return joblib.load("creditwise_model.pkl")

model = load_model()

st.title("CreditWise")
st.subheader("Loan Default Risk Predictor")

st.info(
    "Educational project only. Predictions are estimates, "
    "not lending decisions."
)

with st.form("loan_form"):
    income = st.number_input(
        "Annual income", min_value=1, value=60000
    )
    employment = st.number_input(
        "Employment length (years)", min_value=0.0, value=5.0
    )
    loan_amount = st.number_input(
        "Loan amount", min_value=1, value=12000
    )
    loan_percent = st.number_input(
        "Loan amount as a proportion of income",
        min_value=0.0, max_value=1.0, value=0.20
    )
    credit_history = st.number_input(
        "Credit history length (years)", min_value=0, value=6
    )
    home = st.selectbox(
        "Home ownership", ["RENT", "OWN", "MORTGAGE", "OTHER"]
    )
    purpose = st.selectbox(
        "Loan purpose",
        [
            "PERSONAL", "EDUCATION", "MEDICAL",
            "VENTURE", "HOMEIMPROVEMENT", "DEBTCONSOLIDATION"
        ]
    )
    previous_default = st.selectbox(
        "Previous default on file?", ["N", "Y"]
    )

    submitted = st.form_submit_button("Predict Default Probability")

if submitted:
    borrower = pd.DataFrame([{
        "person_income": income,
        "person_emp_length": employment,
        "loan_amnt": loan_amount,
        "loan_percent_income": loan_percent,
        "cb_person_cred_hist_length": credit_history,
        "person_home_ownership": home,
        "loan_intent": purpose,
        "cb_person_default_on_file": previous_default
    }])

    probability = model.predict_proba(borrower)[0, 1]

    st.metric("Estimated Default Probability", f"{probability:.2%}")
    st.progress(float(probability))
