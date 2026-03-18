import pandas as pd
import nltk
import re
import os
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB

# Download stopwords (only first time)
nltk.download('stopwords')

# ==============================
# 1. Load Dataset
# ==============================

DATA_PATH = "data/dataset.csv"

if not os.path.exists(DATA_PATH):
    raise FileNotFoundError(f"❌ Dataset not found at {DATA_PATH}. Run create_dataset.py first.")

df = pd.read_csv(DATA_PATH)

# ==============================
# 2. Text Cleaning Function
# ==============================

stop_words = set(stopwords.words('english'))

def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'[^a-zA-Z]', ' ', text)
    words = text.split()
    words = [word for word in words if word not in stop_words]
    return " ".join(words)

# Apply cleaning
df['cleaned_text'] = df['text'].apply(clean_text)

# ==============================
# 3. Feature Extraction
# ==============================

vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(df['cleaned_text'])
y = df['sentiment']

# ==============================
# 4. Train Model
# ==============================

model = MultinomialNB()
model.fit(X, y)

print("✅ Model trained successfully!\n")

# ==============================
# 5. Prediction Function
# ==============================

def predict_sentiment(sentence):
    cleaned = clean_text(sentence)
    vector = vectorizer.transform([cleaned])
    prediction = model.predict(vector)
    return prediction[0]

# ==============================
# 6. User Input Loop
# ==============================

while True:
    user_input = input("Enter a sentence (type 'exit' to quit): ")

    if user_input.lower() == "exit":
        print("👋 Exiting program...")
        break

    result = predict_sentiment(user_input)
    print(f"👉 Sentiment: {result}\n")