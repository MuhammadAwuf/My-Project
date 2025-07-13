import streamlit as st
import joblib

# Load model and vectorizer
model = joblib.load("logistic_model.pkl")
vectorizer = joblib.load("tfidf_vectorizer.pkl")

# UI Header
st.title("📰 Fake News Classifier")
st.subheader("Check whether a news article is Fake or Real")

# User Input
user_input = st.text_area("Paste your news article text here")

if st.button("Classify"):
    if user_input.strip() == "":
        st.warning("Please enter some text to classify.")
    else:
        # Vectorize input
        transformed_input = vectorizer.transform([user_input])
        prediction = model.predict(transformed_input)[0]

        # Display result
        if prediction == 1:
            st.success("✅ This news article is Real.")
        else:
            st.error("🚫 This news article is Fake.")
