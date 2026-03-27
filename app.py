import streamlit as st
import pickle
import nltk
import string
import re
import os

from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer


# Setup NLTK data directory
nltk_data_dir = os.path.join(os.getcwd(), "nltk_data")
os.makedirs(nltk_data_dir, exist_ok=True)
nltk.data.path.append(nltk_data_dir)

# Function to safely download resources
def download_nltk_resource(resource):
    try:
        nltk.data.find(resource)
    except LookupError:
        nltk.download(resource.split("/")[-1], download_dir=nltk_data_dir)

# Download all required resources
download_nltk_resource('tokenizers/punkt')
download_nltk_resource('tokenizers/punkt_tab')
download_nltk_resource('corpora/stopwords')
download_nltk_resource('corpora/wordnet')
download_nltk_resource('corpora/omw-1.4')        


# Download once
#nltk.download('punkt', download_dir=nltk_data_dir)

#nltk.download('stopwords', download_dir=nltk_data_dir)
#nltk.download('wordnet', download_dir=nltk_data_dir)
#nltk.download('omw-1.4', download_dir=nltk_data_dir)



# Load model & vectorizer
model = pickle.load(open("Sentiment_analysis.pkl", "rb"))
vectorizer = pickle.load(open("vectorizer.pkl", "rb"))

# Preprocessing setup
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

def preprocess(text):
    text = text.lower()
    text=re.sub(r"\b(not|no|never)\b\s(\w+)",r"\1_\2",text)
    text = text.translate(str.maketrans('', '', string.punctuation))
    tokens = word_tokenize(text)
    tokens = [w for w in tokens if w not in stop_words]
    tokens = [lemmatizer.lemmatize(w) for w in tokens]
    return " ".join(tokens)

# UI
st.title("🍽️ Restaurant Review Classifier")

review = st.text_area("Enter your review:")

if st.button("Predict"):
    if review.strip() == "":
        st.warning("Please enter a review")
    else:
        clean = preprocess(review)
        vector = vectorizer.transform([clean])
        prediction = model.predict(vector)

        if prediction[0] == 1:
            st.success("Positive Review 😊")
        else:
             st.error("Negative Review 😡")
