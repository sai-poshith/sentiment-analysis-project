import streamlit as st
import pickle

# Load model
with open("model.pkl", "rb") as file:
    model = pickle.load(file)

st.title("Sentiment Analysis App")

user_input = st.text_area("Enter Text")

if st.button("Analyze Sentiment"):
    prediction = model.predict([user_input])[0]
    st.success(f"Predicted Sentiment: {prediction}")
