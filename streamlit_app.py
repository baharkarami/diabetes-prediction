import streamlit as st
import numpy as np
import joblib
import  os


BASE_DIR = os.path.dirname(os.path.dirname(__file__))

# -------------------------
# Page Config
# -------------------------
st.set_page_config(
    page_title="Diabetes Prediction AI",
    page_icon="🩺",
    layout="centered"
)

# -------------------------
# Load Models
# -------------------------
models = {
    "Logistic Regression": joblib.load(os.path.join("models", "logistic_regression_model.pkl")),
    "Random Forest": joblib.load(os.path.join("models", "random_forest_model.pkl")),
    "SVM": joblib.load(os.path.join("models", "svm_model.pkl")),
    "Gradient Boosting": joblib.load(os.path.join("models", "gradient_boosting_model.pkl")),
    "XGBoost": joblib.load(os.path.join(BASE_DIR, "models", "xgboost_model.pkl"))
}

# -------------------------
# Header
# -------------------------
st.markdown(
    "<h1 style='text-align: center; color: #2E86C1;'>🩺 Diabetes Prediction System</h1>",
    unsafe_allow_html=True
)

st.markdown(
    "<p style='text-align: center;'>Machine Learning powered medical diagnosis assistant</p>",
    unsafe_allow_html=True
)

st.divider()

# -------------------------
# Sidebar - Model selection
# -------------------------
st.sidebar.title("⚙️ Settings")

model_choice = st.sidebar.selectbox(
    "Choose Model",
    ["All Models"] + list(models.keys())
)

st.sidebar.markdown("---")
st.sidebar.info("Enter patient details to predict diabetes risk.")

# -------------------------
# Inputs (Main UI)
# -------------------------
st.subheader("📋 Patient Information")

col1, col2 = st.columns(2)

with col1:
    pregnancies = st.number_input("Pregnancies", 0, 20, 1)
    glucose = st.number_input("Glucose", 0, 200, 100)
    blood_pressure = st.number_input("Blood Pressure", 0, 150, 70)
    skin_thickness = st.number_input("Skin Thickness", 0, 100, 20)

with col2:
    insulin = st.number_input("Insulin", 0, 900, 80)
    bmi = st.number_input("BMI", 0.0, 70.0, 25.0)
    dpf = st.number_input("Diabetes Pedigree Function", 0.0, 2.5, 0.5)
    age = st.number_input("Age", 1, 100, 30)

user_data = np.array([[pregnancies, glucose, blood_pressure,
                       skin_thickness, insulin, bmi, dpf, age]])

st.divider()

# -------------------------
# Predict Button
# -------------------------
predict_btn = st.button("🔍 Predict Diabetes Risk")

# -------------------------
# Prediction Logic
# -------------------------
if predict_btn:

    st.subheader("📊 Prediction Results")

    if model_choice == "All Models":

        for name, model in models.items():
            pred = model.predict(user_data)[0]

            if hasattr(model, "predict_proba"):
                prob = model.predict_proba(user_data)[0][1] * 100
            else:
                prob = None

            if pred == 1:
                st.error(f"🔴 {name}: Diabetic")
            else:
                st.success(f"🟢 {name}: Not Diabetic")

            if prob is not None:
                st.info(f"Confidence: {prob:.2f}%")

            st.markdown("---")

    else:
        model = models[model_choice]
        pred = model.predict(user_data)[0]

        if hasattr(model, "predict_proba"):
            prob = model.predict_proba(user_data)[0][1] * 100
        else:
            prob = None

        if pred == 1:
            st.error("🔴 Result: Diabetic")
        else:
            st.success("🟢 Result: Not Diabetic")

        if prob is not None:
            st.metric(label="Confidence Score", value=f"{prob:.2f}%")