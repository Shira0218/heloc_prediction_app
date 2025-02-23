#!/usr/bin/env python
# coding: utf-8

# In[1]:


#pip install streamlit


# In[3]:

import streamlit as st
import pandas as pd
import pickle

# Load the trained model
with open('model.pkl', 'rb') as file:
    model = pickle.load(file)

st.title('HELOC Eligibility Prediction App')

# File uploader
uploaded_file = st.file_uploader("Choose a file", type=['csv', 'xlsx'])

if uploaded_file is not None:
    try:
        # Read the file
        if uploaded_file.name.endswith('.csv'):
            df = pd.read_csv(uploaded_file)
        elif uploaded_file.name.endswith('.xlsx'):
            df = pd.read_excel(uploaded_file)
        
        st.write("Uploaded Data Preview:")
        st.write(df.head())
        
        # Perform label encoding on 'RiskPerformance'
        df['RiskPerformance'] = df['RiskPerformance'].map({'Bad': 0, 'Good': 1})
        st.write("After Label Encoding:")
        st.write(df[['RiskPerformance']].head())
        
        # Make predictions
        if 'RiskPerformance' in df.columns:
            input_data = df.drop(columns=['RiskPerformance'], errors='ignore')
        else:
            input_data = df

        predictions = model.predict(input_data)
        
        # Convert predictions to human-readable labels
        results = ['Accept: Good credit performance expected.' if pred == 1 else 'Reject: High risk of poor credit performance.' for pred in predictions]
        
        # Display predictions
        st.write("Predictions:")
        st.write(pd.DataFrame(results, columns=['Result']))

    except Exception as e:
        st.error(f"Error processing the file: {e}")
else:
    st.info("Please upload a CSV or XLSX file.")
