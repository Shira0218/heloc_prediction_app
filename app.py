#!/usr/bin/env python
# coding: utf-8

# In[1]:


#pip install streamlit


# In[3]:

import streamlit as st
import pandas as pd
import numpy as np
import pickle

# Load the trained model
model = pickle.load(open('model.pkl', 'rb'))

# App title
st.title('HELOC Eligibility Prediction App')

# File upload
uploaded_file = st.file_uploader("Choose a file", type=["csv", "xlsx"])

if uploaded_file is not None:
    try:
        # Read the uploaded file
        if uploaded_file.name.endswith('.csv'):
            df = pd.read_csv(uploaded_file)
        else:
            df = pd.read_excel(uploaded_file)

        st.subheader('Data Preview:')
        st.write(df.head())

        # Preprocess data (e.g., Label Encoding)
        if 'RiskPerformance' in df.columns:
            df['RiskPerformance'] = df['RiskPerformance'].map({'Bad': 0, 'Good': 1})
        
        st.subheader('After Label Encoding:')
        st.write(df[['RiskPerformance']].head())

        # Make predictions
        predictions = model.predict(df.drop(columns=['RiskPerformance'], errors='ignore'))

        # Map predictions to meaningful output
        result = ["Accept: The applicant meets the eligibility criteria." 
                  if int(pred) == 1 
                  else "Reject: The applicant does not meet the eligibility criteria." 
                  for pred in predictions]

        # Display predictions with explanations
        st.subheader("Predictions:")
        st.write(pd.DataFrame(result, columns=["Eligibility Status"]))

    except Exception as e:
        st.error(f"Error processing the file: {e}")
else:
    st.info('Please upload a CSV or XLSX file.')
