import streamlit as st
import pickle
import re
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import nltk

# -----------------------------
# Step 1: NLTK setup
# -----------------------------
nltk.download('stopwords')
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

# -----------------------------
# Step 2: Load Model & Vectorizer
# -----------------------------
model = pickle.load(open('../sentiment_model.pkl', 'rb'))
vectorizer = pickle.load(open('../tfidf_vectorizer.pkl', 'rb'))

# -----------------------------
# Step 3: Text Cleaning Function
# -----------------------------
def clean_text(text):
    text = re.sub(r"http\S+|www\S+|https\S+", '', str(text))
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    text = text.lower()
    words = text.split()
    words = [lemmatizer.lemmatize(word) for word in words if word not in stop_words]
    return " ".join(words)

# -----------------------------
# Step 4: Streamlit UI
# -----------------------------
st.set_page_config(page_title="Sentiment Analysis System", layout="wide")
st.title("💬 Sentiment Analysis System")
st.markdown("Enter any sentence or review, and the AI will predict its sentiment (positive, neutral, negative).")

# User input
user_input = st.text_area("Enter your text here:")

if st.button("Predict Sentiment"):
    if user_input.strip() == "":
        st.warning("Please enter some text.")
    else:
        cleaned_input = clean_text(user_input)
        vectorized_input = vectorizer.transform([cleaned_input])
        prediction = model.predict(vectorized_input)[0]
        st.success(f"Predicted Sentiment: **{prediction.capitalize()}**")
        