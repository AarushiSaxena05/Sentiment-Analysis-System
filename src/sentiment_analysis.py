import pandas as pd
import nltk
import re
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB

# Download stopwords
nltk.download('stopwords')

# Load dataset
df = pd.read_csv("data/dataset.csv")

# Text cleaning
stop_words = set(stopwords.words('english'))

def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'[^a-zA-Z]', ' ', text)
    words = text.split()
    words = [word for word in words if word not in stop_words]
    return " ".join(words)

# Apply cleaning
df['cleaned_text'] = df['text'].apply(clean_text)

# Features & labels
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(df['cleaned_text'])
y = df['sentiment']

# Train model
model = MultinomialNB()
model.fit(X, y)

print("✅ Model trained successfully!\n")

# Infinite loop for input
while True:
    user_input = input("Enter a sentence (or type 'exit' to stop): ")

    if user_input.lower() == "exit":
        print("👋 Exiting...")
        break

    cleaned = clean_text(user_input)
    vector = vectorizer.transform([cleaned])
    prediction = model.predict(vector)

    print("👉 Predicted Sentiment:", prediction[0], "\n")