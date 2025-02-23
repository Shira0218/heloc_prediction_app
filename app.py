#!/usr/bin/env python
# coding: utf-8

# In[1]:


#pip install streamlit


# In[3]:


import streamlit as st
import pandas as pd
import numpy as np
import joblib  # For loading the pre-trained model
from sklearn.ensemble import RandomForestClassifier

model = joblib.load('model.pkl')

st.title('HELOC Eligibility Prediction App')

uploaded_file = st.file_uploader('Choose a file', type=['csv', 'xlsx'])
if uploaded_file is not None:
    try:
        if uploaded_file.name.endswith('.csv'):
            df = pd.read_csv(uploaded_file)
        elif uploaded_file.name.endswith('.xlsx'):
            df = pd.read_excel(uploaded_file)

        st.write('Uploaded Data Preview:')
        st.write(df.head())

       
        if 'RiskPerformance' in df.columns:
            df['RiskPerformance'] = df['RiskPerformance'].map({'Bad': 0, 'Good': 1})
            st.write('After Label Encoding:')
            st.write(df['RiskPerformance'].head())

        X = df.drop(columns=['RiskPerformance'], errors='ignore')

        if st.button('Predict Eligibility'):
            predictions = model.predict(X)
            st.write('Predictions:')
            st.write(predictions)
    except Exception as e:
        st.error(f'Error processing the file: {e}')
