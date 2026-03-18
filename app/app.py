import streamlit as st
import pickle
import re
import os
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# -----------------------------
# NLTK setup
# -----------------------------
nltk.download('stopwords')
nltk.download('wordnet')

stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

# -----------------------------
# Load Model & Vectorizer (SAFE PATH)
# -----------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

model_path = os.path.join(BASE_DIR, "..", "sentiment_model.pkl")
vectorizer_path = os.path.join(BASE_DIR, "..", "tfidf_vectorizer.pkl")

model = pickle.load(open(model_path, 'rb'))
vectorizer = pickle.load(open(vectorizer_path, 'rb'))

# -----------------------------
# Text Cleaning Function
# -----------------------------
def clean_text(text):
    text = re.sub(r"http\S+|www\S+|https\S+", '', str(text))
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    text = text.lower()
    words = text.split()
    words = [lemmatizer.lemmatize(word) for word in words if word not in stop_words]
    return " ".join(words)

# -----------------------------
# Streamlit UI
# -----------------------------
st.set_page_config(page_title="Sentiment Analysis System", layout="centered")

st.title("💬 Sentiment Analysis System")
st.write("Type a sentence and get sentiment prediction instantly!")

user_input = st.text_area("Enter your text:")

if st.button("Predict"):
    if user_input.strip() == "":
        st.warning("Please enter some text.")
    else:
        cleaned_input = clean_text(user_input)
        vectorized_input = vectorizer.transform([cleaned_input])
        prediction = model.predict(vectorized_input)[0]

        if prediction.lower() == "positive":
            st.success(f"😊 Sentiment: {prediction.capitalize()}")
        elif prediction.lower() == "negative":
            st.error(f"😠 Sentiment: {prediction.capitalize()}")
        else:
            st.info(f"😐 Sentiment: {prediction.capitalize()}")