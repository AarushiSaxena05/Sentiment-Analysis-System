import streamlit as st
import pickle
import re
import numpy as np
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import nltk

# -----------------------------
# NLTK setup (run once)
# -----------------------------
nltk.download('stopwords')
nltk.download('wordnet')

stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

# -----------------------------
# Load Model & Vectorizer
# -----------------------------
model = pickle.load(open('sentiment_model.pkl', 'rb'))
vectorizer = pickle.load(open('tfidf_vectorizer.pkl', 'rb'))

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
st.set_page_config(page_title="Advanced Sentiment Analysis", layout="centered")

st.title("💬 Advanced Sentiment Analysis System")
st.markdown("Analyze sentiment with confidence scores, breakdown, and insights 🚀")

# Input
user_input = st.text_area("✍️ Enter your text here:")

# Sample text buttons
col1, col2, col3 = st.columns(3)
if col1.button("😊 Positive Example"):
    user_input = "This product is amazing and I love it!"
if col2.button("😡 Negative Example"):
    user_input = "This is the worst experience ever."
if col3.button("😐 Neutral Example"):
    user_input = "The product is okay, not too bad."

# Predict
if st.button("🔍 Analyze Sentiment"):
    if user_input.strip() == "":
        st.warning("⚠️ Please enter some text.")
    else:
        with st.spinner("Analyzing..."):
            cleaned_input = clean_text(user_input)
            vectorized_input = vectorizer.transform([cleaned_input])

            prediction = model.predict(vectorized_input)[0]

            # Probability (if model supports it)
            try:
                proba = model.predict_proba(vectorized_input)[0]
                confidence = np.max(proba) * 100
            except:
                proba = None
                confidence = None

        # -----------------------------
        # Result Display
        # -----------------------------
        st.subheader("📊 Result")

        if prediction == "positive":
            st.success("😊 Positive Sentiment")
        elif prediction == "negative":
            st.error("😡 Negative Sentiment")
        else:
            st.warning("😐 Neutral Sentiment")

        # Confidence score
        if confidence:
            st.write(f"**Confidence Score:** {confidence:.2f}%")

        # Detailed breakdown
        if proba is not None:
            st.subheader("📈 Sentiment Breakdown")
            labels = model.classes_
            breakdown = {labels[i]: float(proba[i]) for i in range(len(labels))}
            st.bar_chart(breakdown)

        # Keywords
        st.subheader("🔑 Important Words")
        keywords = cleaned_input.split()[:10]
        st.write(", ".join(keywords))

        # Cleaned text (for learning/debug)
        with st.expander("🧹 Cleaned Text"):
            st.write(cleaned_input)

# Footer
st.markdown("---")
st.markdown("🚀 Built with Streamlit | Advanced NLP Project")