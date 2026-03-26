import pandas as pd
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
import streamlit as st

# ----------------------------------------------------
# 1. Relative path for dataset (news.csv must be in the same folder as this script)
df = pd.read_csv("news.csv")  # <-- relative path, do NOT use absolute path

# 2. Prepare features and labels
X = df['text']
y = df['label']

# 3. Vectorizer & Model training
vectorizer = TfidfVectorizer(stop_words='english')
X_vec = vectorizer.fit_transform(X)

model = LogisticRegression()
model.fit(X_vec, y)

# 4. Save model and vectorizer
with open("model.pkl", "wb") as f:
    pickle.dump(model, f)
with open("vectorizer.pkl", "wb") as f:
    pickle.dump(vectorizer, f)

# 5. Streamlit app
st.title("Spam News Detection 📰")

user_input = st.text_area("Enter news text:")

if st.button("Predict"):
    if user_input.strip() != "":
        # Load vectorizer and model
        with open("vectorizer.pkl", "rb") as f:
            vec = pickle.load(f)
        with open("model.pkl", "rb") as f:
            mdl = pickle.load(f)
        # Transform input
        user_vec = vec.transform([user_input])
        prediction = mdl.predict(user_vec)[0]
        st.success(f"Prediction: {prediction}")
    else:
        st.warning("Please enter some text.")
