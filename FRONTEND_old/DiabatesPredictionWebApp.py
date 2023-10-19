# -*- coding: utf-8 -*-
"""
Created on Mon Oct  9 14:21:49 2023

@author: Shavini
"""

import numpy as np
import pickle
import streamlit as st

loaded_model = pickle.load(open('E://3rd year//Y3S2//FDM//Mini Project//trained_model.sav', 'rb'))

#creating a function for prediction
def diabetes_prediction(input_data): 
    # Changing the input data to a numpy array
    input_data_as_numpy_array = np.asarray(input_data)

    # Reshape the array as we are predicting for one instance
    input_data_reshaped = input_data_as_numpy_array.reshape(1, -1)
    print(input_data_reshaped)

    # Load the scaler used during training (you should save it during training)
    scaler = pickle.load(open('E://3rd year//Y3S2//FDM//Mini Project//scaler.sav', 'rb'))

    # Standardize the input data using the loaded scaler
    std_data = scaler.transform(input_data_reshaped)
    print(std_data)

    # Make predictions
    prediction = loaded_model.predict(std_data)
    print(prediction)

    if prediction[0] == 0:
        return 'The person is not diabetic'
    else:
        return 'The person is diabetic'    
    
def main():
    
    #giving a title
    st.title('Diabets Prediction Web App')
    
    #getting the input data from the user    
    
    Pregnancies = st.text_input('Number of Pregnancies')
    Glucose = st.text_input('Glucose Level')
    BloodPressure = st.text_input('Blood Pressure Value')
    SkinThickness = st.text_input('Skin Thickness Value')
    Insulin = st.text_input('Insulin Level')
    BMI = st.text_input('BMI Value')
    DiabetesPedigreeFunction = st.text_input('Diabetes Pedigree Function Value')
    Age = st.text_input('Age of the Person')
    
    
    #code for Prediction
    diagnosis = ''
    
    #creating a button for Prediction
    if st.button('Diabetes Test Results'):
        diagnosis = diabetes_prediction([Pregnancies,Glucose,BloodPressure,SkinThickness,Insulin,BMI,DiabetesPedigreeFunction,Age])
        
    st.success(diagnosis)  

if __name__ == '__main__':
    main()