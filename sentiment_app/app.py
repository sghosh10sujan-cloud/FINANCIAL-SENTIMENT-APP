import streamlit as st
import joblib
import json
import string
import nltk
from nltk.corpus import stopwords
import os

BASE_DIR=os.path.dirname(os.path.abspath(__file__)
nltk.download('stopwords')
stop_words = set(stopwords.words('english'))

# Load saved artifacts
model = joblib.load(os.path.join(BASE_DIR, 'sentiment_model.pkl'))
vectorizer = joblib.load(os.path.join(BASE_DIR, 'tfidf_vectorizer.pkl'))

with open(os.path.join(BASE_DIR, 'sentiment_numbers.json')) as f:
    sentiment_numbers = json.load(f)

# reverse mapping: number -> label
number_to_sentiment = {v: k for k, v in sentiment_numbers.items()}

def clean_text(text):
    text = text.lower()
    text = text.translate(str.maketrans('', '', string.punctuation))
    words = text.split()
    words = [w for w in words if w not in stop_words]
    return " ".join(words)

st.title("Financial Sentiment Analyzer")
st.write("Enter a financial news sentence to predict its sentiment (positive / negative / neutral).")

user_input = st.text_area("Sentence:", height=100)

if st.button("Predict"):
    if user_input.strip() == "":
        st.warning("Please enter a sentence.")
    else:
        cleaned = clean_text(user_input)
        vectorized = vectorizer.transform([cleaned])
        prediction = model.predict(vectorized)[0]
        prediction_label = number_to_sentiment[prediction]
    
        probs = model.predict_proba(vectorized)[0]

        st.subheader(f"Prediction: **{prediction_label.upper()}**")
        st.write("Confidence per class:")
        for idx, prob in enumerate(probs):
            st.write(f"{number_to_sentiment[idx]}: {prob:.2%}")
