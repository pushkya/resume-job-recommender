# -*- coding: utf-8 -*-
"""
Created on Mon Jul 26 20:58:36 2021

@author: Pushkar
"""
import streamlit as st
import process as pda
import pandas as pd
import pca as pc
import matplotlib.pyplot as plt
import word_sim
import pickle
#st.set_option('deprecation.showPyplotGlobalUse', False)
import re

#Introduce App
st.title('Job Recommender')
st.markdown('(Best fit Job field based on your resume)')
st.sidebar.markdown("See which jobs best match your profile and optimize your resume / LinkedIn!")

#Get and transform user's resume or linkedin
user_input = st.text_area("copy and paste your resume or linkedin here", '')

user_input = str(user_input)
user_input = re.sub('[^a-zA-Z0-9\.]', ' ', user_input)
user_input = user_input.lower()

user_input = pd.Series(user_input)

#load NLP + classification models

topic_model = pickle.load(open('topic_model.sav', 'rb'))
classifier = pickle.load(open('classification_model1.sav', 'rb'))
vec = pickle.load(open('job_vec.sav', 'rb'))

classes, prob = pda.main(user_input, topic_model, classifier, vec)

data = pd.DataFrame(zip(classes.T, prob.T), columns = ['jobs', 'probability'])

#Plot probability of person belonging to a job class
def plot_user_probability():
    fig = plt.figure()
    ax = fig.add_subplot()
    ax.barh(data['jobs'], data['probability'], color = 'r')
    ax.set_title('Percent Match of Job Type')
    st.write(fig)

#Plot where user fits in with other job clusters
def plot_clusters():
    st.markdown('This chart uses PCA to show you where you fit among the different job archetypes.')
    X_train, pca_train, y_train, y_vals, pca_model = pc.create_clusters()
    for i, val in enumerate(y_train.unique()):
        y_train = y_train.apply(lambda x: i if x == val else x)
    example = user_input
    doc = pc.transform_user_resume(pca_model, example)

    pc.plot_PCA_2D(pca_train, y_train, y_vals, doc)
    st.pyplot()

plot_user_probability()
st.title('Representation Among Job Types')
plot_clusters()

st.title('Find Matching Keywords')
st.markdown('This function shows you which keywords your resume either contains or doesnt contain, according to the most significant words in each job description.')
st.markdown("The displayed keywords are stemmed, ie 'analysis' --> 'analys' and 'commision' --> 'commiss'")
option = st.selectbox(
    'Which job would you like to compare to?',
 ('ux,designer', 'data,analyst', 'project,manager', 'product,manager', 'account,manager', 'consultant', 'marketing', 'sales',
 'data,scientist', 'full,stack','big data engineer','artificial intelligence, deep learning', 'data quality manager', 'data visualisation expert'))

st.write('You selected:', option)
matches, misses = word_sim.resume_reader(user_input, option)
match_string = ' '.join(matches)
misses_string = ' '.join(misses)

st.markdown('Matching Words:')
st.markdown(match_string)
st.markdown('Missing Words:')
st.markdown(misses_string)
