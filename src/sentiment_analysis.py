import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report
import pickle
import os

# -----------------------------
# Step 1: NLTK Downloads
# -----------------------------
nltk.download('stopwords')
stop_words = set(stopwords.words('english'))

# -----------------------------
# Step 2: Text Preprocessing
# -----------------------------
def clean_text(text):
    """
    Clean the text data:
    - Lowercase
    - Remove URLs
    - Remove special characters and numbers
    - Remove stopwords
    """
    text = re.sub(r"http\S+|www\S+|https\S+", '', str(text))
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    text = text.lower()
    words = text.split()
    words = [word for word in words if word not in stop_words]
    return " ".join(words)

# -----------------------------
# Step 3: Load Dataset
# -----------------------------
# Change the path if your dataset is somewhere else
DATA_PATH = '../data/dataset.csv'  

if not os.path.exists(DATA_PATH):
    raise FileNotFoundError(f"Dataset not found at {DATA_PATH}. Please place your dataset in the data folder.")

df = pd.read_csv(DATA_PATH)

# Check for required columns
if 'text' not in df.columns or 'sentiment' not in df.columns:
    raise KeyError("Dataset must have 'text' and 'sentiment' columns.")

# Preprocess text
df['cleaned_text'] = df['text'].apply(clean_text)

# -----------------------------
# Step 4: Feature Extraction
# -----------------------------
vectorizer = TfidfVectorizer(max_features=5000)
X = vectorizer.fit_transform(df['cleaned_text'])
y = df['sentiment']

# Save vectorizer for reuse
pickle.dump(vectorizer, open('tfidf_vectorizer.pkl', 'wb'))

# -----------------------------
# Step 5: Train-Test Split
# -----------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# -----------------------------
# Step 6: Train Model
# -----------------------------
model = MultinomialNB()
model.fit(X_train, y_train)

# -----------------------------
# Step 7: Evaluate Model
# -----------------------------
y_pred = model.predict(X_test)
print("Accuracy on Test Set:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# Save trained model
pickle.dump(model, open('sentiment_model.pkl', 'wb'))

# -----------------------------
# Step 8: Test Custom Sentences
# -----------------------------
def predict_sentiment(sentence):
    cleaned = clean_text(sentence)
    vector = vectorizer.transform([cleaned])
    prediction = model.predict(vector)[0]
    return prediction

# Example usage
while True:
    user_input = input("\nEnter a sentence (or type 'exit' to quit): ")
    if user_input.lower() == 'exit':
        break
    sentiment = predict_sentiment(user_input)
    print("Predicted Sentiment:", sentiment)