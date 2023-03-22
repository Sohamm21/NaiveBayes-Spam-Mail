import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
import streamlit as st

st.title("Email Spam using Naive Bayes")
url = 'https://raw.githubusercontent.com/Sohamm21/NaiveBayes-Spam-Mail/main/spam.csv';
df = pd.read_csv(url, encoding="ISO-8859-1")
st.write(df)
df = df.rename(columns={'v1': 'class', 'v2': 'sms'})
df.drop(['Unnamed: 2', 'Unnamed: 3', 'Unnamed: 4'], axis='columns', inplace=True)

# converting categorical data to numerical data
df['spam'] = df['class'].apply(lambda x:1 if x=='spam' else 0)

df.drop('class', axis='columns', inplace=True)

X_train, X_test, y_train, y_test = train_test_split(df.sms, df.spam)

clf = Pipeline([
    ('vectorizer', CountVectorizer()),
    ('nb', MultinomialNB())
])

# train the model
clf.fit(X_train, y_train)
score = clf.score(X_test, y_test) * 100

st.write('The accuracy is:', score)

email = st.text_input("Enter an email message")

# Make a prediction using the machine learning model
if email:
    result = clf.predict([email])
    if result[0] == 0:
        st.write('Not a spam message')
    else:
        st.write('Spam message!')
