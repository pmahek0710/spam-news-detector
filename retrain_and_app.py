# retrain_and_app.py
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
import pickle
import streamlit as st

# -------------------------------
# 1️⃣ LOAD DATASET
# -------------------------------
st.title("Spam News Detection")  # Your app title

# Update path to your Downloads folder
DATA_PATH = r"C:\Users\pmahe\Downloads\news.csv"

try:
    df = pd.read_csv(DATA_PATH)
    st.success(f"Dataset loaded! Rows: {len(df)}")
except FileNotFoundError:
    st.error(f"Dataset not found! Make sure {DATA_PATH} exists.")
    st.stop()

# -------------------------------
# 2️⃣ CLEAN DATA
# -------------------------------
# Keep only text and label
df = df[['text', 'label']].dropna()
df['label'] = df['label'].str.strip().str.upper()  # Ensure uniform labels

# -------------------------------
# 3️⃣ VECTORIZE & TRAIN MODEL
# -------------------------------
vectorizer = TfidfVectorizer(
    max_features=10000,
    stop_words='english',
    ngram_range=(1,2)
)

X = vectorizer.fit_transform(df['text'])
y = df['label']

model = LogisticRegression(max_iter=1000)
model.fit(X, y)

# Save model & vectorizer
pickle.dump(model, open(r"C:\Users\pmahe\Downloads\model.pkl", "wb"))
pickle.dump(vectorizer, open(r"C:\Users\pmahe\Downloads\vectorizer.pkl", "wb"))

st.success("✅ Model trained and saved successfully!")

# -------------------------------
# 4️⃣ STREAMLIT APP FOR TESTING
# -------------------------------
st.subheader("Test Your News")

news_input = st.text_area("Enter news text here:")

if st.button("Check News"):
    if news_input.strip() == "":
        st.warning("Please enter some news text!")
    else:
        # Load saved model & vectorizer
        model = pickle.load(open(r"C:\Users\pmahe\Downloads\model.pkl", "rb"))
        vectorizer = pickle.load(open(r"C:\Users\pmahe\Downloads\vectorizer.pkl", "rb"))

        news_vector = vectorizer.transform([news_input])
        pred = model.predict(news_vector)[0]

        label = "REAL" if pred == "REAL" else "FAKE"
        st.write(f"Prediction: 🚨 {label} News!")