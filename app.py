import streamlit as st
import pickle
import re

model = pickel.load(open('knn.pkl','rb'))
tfid = pickle.load(open('tfidf.pkl','rb'))

st.title("Real Time Resume Screeing Web")