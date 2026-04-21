import streamlit as st
import numpy as np
import pandas as pd
from sklearn.datasets import load_diabetes
from sklearn.linear_model import LinearRegression

# ---- Page Config ----
st.set_page_config(
    page_title="Diabetes Progression Prediction",
    page_icon="🩺",
    layout="wide"
)

# ---- Load Dataset ----
data = load_diabetes()
X = data.data
y = data.target
feature_names = data.feature_names

# Convert to DataFrame (for understanding)
df = pd.DataFrame(X, columns=feature_names)

# ---- Train Model ----
model = LinearRegression()
model.fit(X, y)

# ---- Title ----
st.title("🩺 Diabetes Progression Prediction App")
st.markdown("Predict diabetes disease progression based on medical attributes.")

st.divider()

# ---- Layout ----
col1, col2 = st.columns([1, 1])

# ---- Input Section ----
with col1:
    st.subheader("📏 Enter Patient Details")

    age = st.slider("Age", -0.2, 0.2, 0.0)
    sex = st.slider("Sex", -0.2, 0.2, 0.0)
    bmi = st.slider("BMI", -0.2, 0.2, 0.0)
    bp = st.slider("Blood Pressure", -0.2, 0.2, 0.0)
    s1 = st.slider("S1 (TC)", -0.2, 0.2, 0.0)
    s2 = st.slider("S2 (LDL)", -0.2, 0.2, 0.0)
    s3 = st.slider("S3 (HDL)", -0.2, 0.2, 0.0)
    s4 = st.slider("S4 (TCH)", -0.2, 0.2, 0.0)
    s5 = st.slider("S5 (LTG)", -0.2, 0.2, 0.0)
    s6 = st.slider("S6 (GLU)", -0.2, 0.2, 0.0)

    predict_button = st.button("🔍 Predict Progression")

# ---- Output Section ----
with col2:
    st.subheader("📊 Prediction Result")

    if predict_button:
        input_data = np.array([[age, sex, bmi, bp, s1, s2, s3, s4, s5, s6]])

        prediction = model.predict(input_data)

        st.success(f"### 📈 Predicted Disease Progression Score: {prediction[0]:.2f}")

        st.info("Higher value indicates higher disease progression.")
    else:
        st.info("Enter details and click **Predict Progression**")

# ---- Footer ----
st.divider()
st.caption("Built with ❤️ using Streamlit")
