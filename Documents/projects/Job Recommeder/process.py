# -*- coding: utf-8 -*-
"""
Created on Thu Jul 22 13:35:22 2021

@author: Pushkar
"""
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.tokenize import TreebankWordTokenizer
from nltk.stem import PorterStemmer
from sklearn.decomposition import TruncatedSVD

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.metrics import accuracy_score
import pickle


def tokenize_stem(series):

    tokenizer =TreebankWordTokenizer()
    stemmer = PorterStemmer()
    series = series.apply(lambda x: x.replace("\n", ' '))
    series = series.apply(lambda x: tokenizer.tokenize(x))
    series = series.apply(lambda x: [stemmer.stem(w) for w in x])
    series = series.apply(lambda x: ' '.join(x))
    return series

def display_topics(model, feature_names, no_top_words, topic_names=None):
    '''
    displays topics and returns list of toppics
    '''

    topic_list = []
    for i, topic in enumerate(model.components_):
        if not topic_names or not topic_names[i]:
            print("\nTopic ", i)
        else:
            print("\nTopic: '",topic_names[i],"'")

        print(", ".join([feature_names[k]
                       for k in topic.argsort()[:-no_top_words - 1:-1]]))
        topic_list.append(", ".join([feature_names[k]
                       for k in topic.argsort()[:-no_top_words - 1:-1]]))
    return model.components_, topic_list

def return_topics(series, num_topics, no_top_words, model, vectorizer):
    '''
    returns document_topic matrix and topic modeling model
    '''
    #turn job into series
    series = tokenize_stem(series)
    #transform series into corpus
    ex_label = [e[:30]+"..." for e in series]
    #set vectorizer ngrams = (2,2)
    vec = vectorizer(stop_words = 'english')

    doc_word = vec.fit_transform(series)

    #build model
    def_model = model(num_topics)
    def_model = def_model.fit(doc_word)
    doc_topic = def_model.transform(doc_word)
    model_components, topic_list = display_topics(def_model, vec.get_feature_names(), no_top_words)
    return def_model.components_, doc_topic, def_model, vec, topic_list#, topics


def process_data():
    '''
    uses the functions above to read in files, model, and return a topic_document dataframe
    '''
    #read in jobs file and get descriptions
    jobs_df = pd.read_csv('newdata.csv')
    jobs_df = jobs_df.dropna()
    jobs_df = jobs_df.reset_index(drop = True)

    array, doc, topic_model, vec, topic_list  = return_topics(jobs_df['description'],20, 10, TruncatedSVD, TfidfVectorizer)

    topic_df = pd.DataFrame(doc)
    topic_df.columns = ['Topic ' + str(i+1) for i in range(len(topic_df.columns)) ]

    topic_df['position'] = jobs_df.position
    topic_df.to_csv('topic_df.csv')
    return topic_df, topic_model, vec, topic_list

def predictive_modeling(df):
    '''
    fits, optimizes, and predicts job class based on topic modeling corpus
    '''
    X,y = df.iloc[:,0:-1], df.iloc[:, -1]
    X_tr, X_te, y_tr, y_te = train_test_split(X,y)

    #param_grid = {'n_estimators': [100,300, 400, 500, 600], 'max_depth': [3,7,9, 11]}
    #rfc = RandomForestClassifier(n_estimators = 500, max_depth = 9)
    #rfc.fit(X_tr, y_tr)
    lo_re = LogisticRegression(n_jobs=3, C=1e5, max_iter=1000)
    lo_re.fit(X_tr, y_tr)
    print('acc: ', np.mean(cross_val_score(lo_re, X_tr, y_tr, scoring = 'accuracy', cv=5)))
    print('test_acc: ', accuracy_score(y_te, lo_re.predict(X_te)))
    print(lo_re.predict(X_te))
    return lo_re

def predict_resume(topic_model, model, resume):
    '''
    transforms a resume based on the topic modeling model and return prediction probabilities per each job class
    '''
    doc = topic_model.transform(resume)
    return model.predict_proba(doc), model.classes_

def get_topic_classification_models():
    jobs_df, model, vec , topic_list= process_data()
    model_1 = predictive_modeling(jobs_df)
    return model, model_1, vec

topic_model, classifier1, vec= get_topic_classification_models()
#topic_model_name = 'topic_model.sav'
classifier_name1 = 'classification_model1.sav'
#vec_name = 'job_vec.sav'
#pickle.dump(topic_model, open(topic_model_name, 'wb'))
pickle.dump(classifier1, open(classifier_name1, 'wb'))
#pickle.dump(vec, open(vec_name, 'wb'))

def main(resume, topic_model, predictor, vec):
    '''
    run code that predicts resume
    '''
    doc = tokenize_stem(resume)
    doc = vec.transform(doc)
    probabilities, classes = predict_resume(topic_model, predictor, doc)
    return classes, probabilities[0]*100
