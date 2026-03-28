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


<img width="976" height="395" alt="load data" src="https://github.com/user-attachments/assets/6144fe8e-3a24-485b-bd01-0c52b7673881" />



### Text Preprocessing

Text preprocessing is crucial for NLP tasks. A custom function named preprocess() is applied to clean the text. The function has following steps: 

Lowercasing – Convert all text to lowercase.

Remove punctuation – Strip punctuation marks.

Tokenization – Split text into individual words using NLTK's word_tokenize.

Stopwords removal – Remove common English stopwords.

Lemmatization – Convert words to their base forms using WordNet lemmatizer.


<img width="1237" height="280" alt="before_func" src="https://github.com/user-attachments/assets/ee0344a2-9999-4a59-8d0f-916cc21084b5" />




<img width="1045" height="306" alt="preprocess_function" src="https://github.com/user-attachments/assets/00312196-e48b-4681-8ddc-4938f1b7ce61" />



<img width="916" height="103" alt="apply_func" src="https://github.com/user-attachments/assets/99bc122b-c3ff-4f51-b02e-85e56cbb9366" />





### Feature Extraction (TF-IDF)


<img width="1042" height="216" alt="before tfidf" src="https://github.com/user-attachments/assets/082f0f86-4f59-4661-88ae-78f7c1929ee9" />


Text data is converted into numerical features using TF-IDF Vectorization: 


<img width="890" height="105" alt="TFIDF" src="https://github.com/user-attachments/assets/e1b19d15-ba93-44e8-b79f-0fd62d1c3ff5" />


### Train-Test Split

The dataset is split into training and testing sets


### Model Training

First I chose Logistic Regression as the classifier.

Import the Logistic Regression class from scikit-learn and trains a Logistic Regression model on the training data to classify text reviews as positive or negative.

<img width="919" height="321" alt="train_lgr" src="https://github.com/user-attachments/assets/5f6ccaf9-3e95-4fce-a10d-d59162578419" />




### Predict
Use my trained model (lgr_model) to predict the outcomes for the test data (X_test), and store those predictions in Y_pred.”


<img width="750" height="60" alt="pred_lgr" src="https://github.com/user-attachments/assets/1931dd3c-99b0-4992-b174-c71c231c464b" />


### Evaluate Logistic Regression model


<img width="1171" height="169" alt="accuracy_lgr" src="https://github.com/user-attachments/assets/e1dde99c-a64f-4015-b0ef-e6db6e52bc63" />


Your model correctly predicted 79% of all cases. Good, but accuracy alone can be misleading — that’s why the confusion matrix matters.


<img width="1229" height="984" alt="cm_lgr" src="https://github.com/user-attachments/assets/d3258ef6-31fa-41a6-b0d0-552555d6e7a7" />




97 → Correctly predicted class 0 (True Negatives)

10 → Incorrectly predicted 1 when it was 0 (False Positives)

32 → Missed class 1 (False Negative)

61 → Correctly predicted class 1 (True Positives)



<img width="1258" height="362" alt="cm1_lgr" src="https://github.com/user-attachments/assets/bc002e5a-c430-4a36-aaca-9218cca8bd55" />







Class 0:

Precision = 0.75

→ When model predicts 0, it's correct 75% of the time

Recall = 0.91


→ It finds 91% of all actual 0s (very good)

F1 = 0.82

→ Balanced performance

Class 1:

Precision = 0.86

→ When it predicts 1, it's usually right 

Recall = 0.66 

→ It only finds 66% of actual 1s


Overall Interpretation

This model is:

Good at identifying class 0

Decent precision for class 1

But misses too many class 1 cases

### Model Training use Naive Bayes Classifier

For getting better results I want to train the data with another model. I chose Naive Bayes Classifier.

Training Multinomial Naive Bayes model.

<img width="1139" height="338" alt="train_nb" src="https://github.com/user-attachments/assets/5734a0be-df68-440f-b7ff-ed26642c3e36" />


using trained model (NB_model) to predict the outcomes for the test data (X_test), and store those predictions in Y_pred.

<img width="1130" height="72" alt="pred_nb" src="https://github.com/user-attachments/assets/f431a4f2-72e9-4a0d-87ce-299b1540cb95" />



### Evaluate Naive bayes model 


<img width="1120" height="126" alt="accracy_nb" src="https://github.com/user-attachments/assets/2a53d1fa-09e3-4805-835a-9e12646858e2" />



Accuracy - model correctly predicted 79% of all cases.


<img width="1072" height="347" alt="CM1" src="https://github.com/user-attachments/assets/f1f69e2a-496d-443f-af84-2656e94afd6c" />








<img width="1050" height="989" alt="CM" src="https://github.com/user-attachments/assets/58c43422-3d36-4a58-a4b0-e280f5910962" />










Class-wise Performance

Class 0:

Precision = 0.84 (better than logistic regression model)

Recall = 0.70 (worse than logistic regression model)

Interpretation:

When it predicts 0, it's usually correct
But it misses many actual 0s

Class 1:
Precision = 0.71  (worse)
Recall = 0.85 (much better!)

### comparing the two model's performance 

Logistic Regression recall for class 1 = 0.66

Naive Bayes recall for class 1 = 0.85






Naive Bayes detects more positive sentiments but makes more mistakes (false positives)

Logistic Regression model is more accurate overall and balanced performance. More trustworthy predictions.

Typical sentiment analysis goal:

want reliable predictions

Not too many false alarms

precision + overall balance matters more.
So I chose Logistic Regression Model is applied to this sentiment Analysis Project.




### Saving the Model

The trained model and TF-IDF vectorizer are saved using pickle.
This allows you to load them later for making predictions without retraining. 

<img width="1272" height="273" alt="pickle" src="https://github.com/user-attachments/assets/4930e259-951d-4519-a523-7dc1510881a7" />


pickle is a built-in Python module used to serialize objects.Convert your model into a file
So you can load it later without retraining


This saves trained Logistic Regression model.



Model expects input in the same format.
The vectorizer converts raw text → numerical features

For sentiment analysis:

Input text → vectorizer.transform()

Transformed data → model.predict()

Need both for real-world use

