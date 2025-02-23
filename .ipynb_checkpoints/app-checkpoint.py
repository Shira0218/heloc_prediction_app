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

# Set the page title
st.title("HELOC Eligibility Prediction App")

# File uploader to accept both CSV and Excel files
uploaded_file = st.file_uploader("Choose a file", type=["csv", "xlsx"])

if uploaded_file is not None:
    # Load CSV or Excel file based on the file type
    if uploaded_file.name.endswith(".csv"):
        data = pd.read_csv(uploaded_file)
    elif uploaded_file.name.endswith(".xlsx"):
        data = pd.read_excel(uploaded_file)
    
    st.write("Uploaded Data Preview:")
    st.dataframe(data.head())

    # Load the pre-trained model (assuming the model is saved as 'model.pkl')
    model = joblib.load('model.pkl')

    # Button to trigger prediction
    if st.button("Predict Eligibility"):
        # Making predictions using the loaded model
        predictions = model.predict(data)
        data['Prediction'] = predictions
        
        st.write("Prediction Results:")
        st.dataframe(data)
        
        # Provide explanations for the predictions
        st.write("Prediction Explanations:")
        for i, pred in enumerate(predictions):
            if pred == 1:
                st.write(f"Row {i+1}: Application Approved ")
            else:
                st.write(f"Row {i+1}: Application Denied - Consider improving credit score or increasing income")

else:
    st.write("Please upload a valid CSV or Excel file to proceed.")


# In[ ]:




