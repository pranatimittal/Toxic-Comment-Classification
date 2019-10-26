import os
import csv
import pandas as pd
import numpy as np

import nltk
from nltk.corpus import stopwords
from nltk.stem.snowball import SnowballStemmer

import sys

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer

def preprocess(data_path = "/Users/sanja/OneDrive/Documents/Python Scripts/Abusive Text Detector/DataSet/train.csv"):


    data_raw = pd.read_csv(data_path)
    data = data_raw.loc[np.random.choice(data_raw.index, size=1200)]
    print("Number of rows in data =",data.shape[0])
    print("Number of columns in data =",data.shape[1])

    data['comment_text'] = data['comment_text'].str.lower()
    data['comment_text'] = data['comment_text'].apply(cleanPunc)
    data['comment_text'] = data['comment_text'].apply(extractAlpha)
    data['comment_text'] = data['comment_text'].apply(stemming)

    #spliting 
    train, test = train_test_split(data, random_state=42, test_size=0.20)

    train_text = train['comment_text']
    test_text = test['comment_text']

    vectorizer = TfidfVectorizer(strip_accents='unicode', norm='l2',
                                analyzer='word', ngram_range=(1,3), stop_words='english')



    features_train = vectorizer.fit_transform(train_text)
    labels_train = train.drop(labels = ['id','comment_text'], axis=1)
    features_test = vectorizer.transform(test_text)
    labels_test = test.drop(labels = ['id','comment_text'], axis=1)

    return features_train, features_test, labels_train, labels_test

#remove punctuation from text
def cleanPunc(s):
    import string
    s = s.replace("\n"," ")
    s = s.translate(str.maketrans('', '', string.punctuation))
    return s

#keeping only alphabets
def extractAlpha(s):
    from string import ascii_letters
    return "".join([ch for ch in s if ch in (ascii_letters + " ")])

#stemming
def stemming(s):
    from nltk.stem.snowball import SnowballStemmer
    stemmer = SnowballStemmer("english")
    comment = ""
    for w in s.split():
        comment += " " + stemmer.stem(w)
    return comment.strip()
