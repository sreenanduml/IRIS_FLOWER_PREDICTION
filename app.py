
import streamlit as st
import joblib
import numpy as np

st.title("🌼 Iris Flower Prediction App")
st.write("Enter flower measurements to predict the species.")

model = joblib.load("iris_model.pkl")

sepal_length = st.number_input("Sepal Length (cm)", min_value=0.0, max_value=10.0, step=0.1)
sepal_width = st.number_input("Sepal Width (cm)", min_value=0.0, max_value=10.0, step=0.1)
petal_length = st.number_input("Petal Length (cm)", min_value=0.0, max_value=10.0, step=0.1)
petal_width = st.number_input("Petal Width (cm)", min_value=0.0, max_value=10.0, step=0.1)

if st.button("Predict"):
    input_data = np.array([[sepal_length, sepal_width, petal_length, petal_width]])
    prediction = model.predict(input_data)[0]

    species = ["Setosa", "Versicolor", "Virginica"]
    st.success(f"Predicted Species: **{species[prediction]}**")
