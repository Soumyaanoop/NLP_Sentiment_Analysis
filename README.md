# NLP_Sentiment_Analysis
This project builds a sentiment analysis model that classifies text reviewsof a restaurant as Positive or Negative. It uses NLTK for text preprocessing, TF-IDF for feature extraction and Logistic Regression for classification. This project is deployed as an interactive web application using Streamlit.

## Overview

The pipeline consists of the following steps:

Load dataset – Load CSV data containing reviews and their labels.
Preprocess text – Clean, tokenize, remove stopwords, and lemmatize text.
Feature extraction – Convert text into numerical features using TF-IDF.
Train model – Train a Logistic Regression classifier.
Save model – Store the trained model and vectorizer for future predictions.

## Technologies Used

Python
Pandas
NLTK (Natural Language Toolkit)
Scikit-learn
Pickle
Streamlit

## Dataset

The dataset (Restuarant_reviews.tsv) contain the following columns:

Review – Text of the user review.
Label – Sentiment of the review (Positive or Negative).


## Workflow

### Import Libraries

The script begins by importing required libraries for:

Data handling (pandas)
Text preprocessing (nltk, string)
Machine learning (sklearn)
Model saving (pickle)

### Download NLP Resources

NLTK resources are downloaded:

Tokenizer (punkt)

Stopwords

Lemmatizer data (wordnet, omw-1.4)

### Load Dataset


### Text Preprocessing

Text preprocessing is crucial for NLP tasks. A custom function named preprocess() is applied to clean the text. The function has following steps: 

Lowercasing – Convert all text to lowercase.

Remove punctuation – Strip punctuation marks.

Tokenization – Split text into individual words using NLTK's word_tokenize.

Stopwords removal – Remove common English stopwords.

Lemmatization – Convert words to their base forms using WordNet lemmatizer.




### Feature Extraction (TF-IDF)

Text data is converted into numerical features using TF-IDF Vectorization: 



### Train-Test Split

The dataset is split into training and testing sets


### Model Training

First I chose Logistic Regression is as the classifier.
Import the Logistic Regression class from scikit-learn and trains a Logistic Regression model on the training data to classify text reviews as positive or negative.



### 




























### Saving the Model
The trained model and TF-IDF vectorizer are saved using pickle.
This allows you to load them later for making predictions without retraining. 





