"""Simple Streamlit UI that calls the FastAPI /predict endpoint."""
import os
import requests
import streamlit as st

API_URL = os.getenv("API_URL", "http://127.0.0.1:8000").rstrip("/")

st.set_page_config(page_title="Adult Income Classifier (Educational)", page_icon="📊", layout="centered")
st.title("Adult Income Classification")
st.caption("Course demonstration: scikit-learn baseline + AutoML + FastAPI")
st.warning("Educational demonstration only. This model is based on historical census data and must not be used for hiring, lending, eligibility, or other decisions about people.")

try:
    info_response = requests.get(f"{API_URL}/model-info", timeout=5)
    info_response.raise_for_status()
    model_info = info_response.json()
    metrics = model_info.get("test_metrics", {})
    st.subheader("Model test performance")
    c1, c2, c3 = st.columns(3)
    c1.metric("Accuracy", f"{metrics.get('accuracy', 0):.3f}")
    c2.metric("F1 score", f"{metrics.get('f1', 0):.3f}")
    c3.metric("ROC-AUC", f"{metrics.get('roc_auc', 0):.3f}")
    st.write("Selected model:", model_info.get("model_name", "Unknown"))
    st.subheader("Top 5 feature importance")
    top_features = model_info.get("top_features", [])
    if top_features:
        st.dataframe(top_features, use_container_width=True)
    else:
        st.info("Feature importance is unavailable until training completes.")
except Exception:
    model_info = None
    st.info("The API is not available. Start it in another terminal with: uvicorn api.main:app --reload")

st.divider()
st.subheader("Enter example feature values")
with st.form("adult_income_form"):
    col1, col2 = st.columns(2)
    with col1:
        age = st.number_input("Age", min_value=17, max_value=100, value=35)
        workclass = st.selectbox("Workclass", ["Private", "Self-emp-not-inc", "Self-emp-inc", "Federal-gov", "Local-gov", "State-gov", "Without-pay", "Never-worked"])
        fnlwgt = st.number_input("fnlwgt", min_value=1, value=150000)
        education = st.selectbox("Education", ["HS-grad", "Some-college", "Bachelors", "Masters", "Doctorate", "Assoc-voc", "Assoc-acdm", "11th", "12th", "10th", "9th", "7th-8th", "5th-6th", "1st-4th", "Preschool", "Prof-school"])
        education_num = st.number_input("Education number", min_value=1, max_value=16, value=9)
        marital_status = st.selectbox("Marital status", ["Never-married", "Married-civ-spouse", "Divorced", "Separated", "Widowed", "Married-spouse-absent", "Married-AF-spouse"])
        occupation = st.selectbox("Occupation", ["Sales", "Exec-managerial", "Prof-specialty", "Adm-clerical", "Craft-repair", "Other-service", "Tech-support", "Machine-op-inspct", "Transport-moving", "Handlers-cleaners", "Farming-fishing", "Protective-serv", "Priv-house-serv", "Armed-Forces"])
    with col2:
        relationship = st.selectbox("Relationship", ["Not-in-family", "Husband", "Wife", "Own-child", "Unmarried", "Other-relative"])
        race = st.selectbox("Race", ["White", "Black", "Asian-Pac-Islander", "Amer-Indian-Eskimo", "Other"])
        sex = st.selectbox("Sex", ["Male", "Female"])
        capital_gain = st.number_input("Capital gain", min_value=0, value=0)
        capital_loss = st.number_input("Capital loss", min_value=0, value=0)
        hours_per_week = st.number_input("Hours per week", min_value=1, max_value=100, value=40)
        native_country = st.selectbox("Native country", ["United-States", "India", "Mexico", "Canada", "Germany", "Philippines", "England", "China", "Other"])
    submitted = st.form_submit_button("Predict income class")

if submitted:
    payload = {
        "age": int(age), "workclass": workclass, "fnlwgt": int(fnlwgt),
        "education": education, "education_num": int(education_num),
        "marital_status": marital_status, "occupation": occupation,
        "relationship": relationship, "race": race, "sex": sex,
        "capital_gain": int(capital_gain), "capital_loss": int(capital_loss),
        "hours_per_week": int(hours_per_week), "native_country": native_country,
    }
    try:
        response = requests.post(f"{API_URL}/predict", json=payload, timeout=30)
        response.raise_for_status()
        result = response.json()
        st.success(f"Predicted class: {result['predicted_income_class']}")
        if result.get("probability_over_50k") is not None:
            st.metric("Model probability for >50K", f"{result['probability_over_50k']:.1%}")
        st.json(result)
    except requests.RequestException as exc:
        st.error(f"Could not contact the API at {API_URL}. Start FastAPI and try again. Details: {exc}")
