import streamlit as st
import pickle
import numpy as np

# Load the trained encoder and model
@st.cache_resource
def load_models():
    with open(r"C:\Users\Ranjan kumar pradhan\.vscode\encoder.pkl", "rb") as e_file:
        encoder = pickle.load(e_file)
    with open(r"C:\Users\Ranjan kumar pradhan\.vscode\Brain_Tumor_Prediction_Model.pkl", "rb") as m_file:
        model = pickle.load(m_file)
    return encoder, model

encoder, model = load_models()

# Title
st.title("🧠 Machine Learning Classification API with Streamlit")

# User Input
st.header("🔢 Enter Features for Prediction")

# Example input fields (modify based on dataset)
feature1 = st.number_input("Feature 1 (Numeric)", value=0.0)
feature2 = st.selectbox("Feature 2 (Categorical)", ["A", "B", "C"])
feature3 = st.slider("Feature 3 (Range)", min_value=0, max_value=100, value=50)

# Convert categorical input using the encoder
encoded_feature2 = encoder.transform([[feature2]])[0]

# Create feature array for prediction
features = np.array([[feature1, encoded_feature2, feature3]])

# Predict
if st.button("🚀 Predict"):
    prediction = model.predict(features)
    st.success(f"🎯 Prediction: {prediction[0]}")
