# -*- coding: utf-8 -*-

import numpy as np 
import pickle
import streamlit as st 
from PIL import Image 
import base64   

css = """
<link rel="stylesheet" href="D:\Desktop\FDM_MiniProject\FDM_MiniProject\FRONTEND\styles.css">
"""
st.markdown(css, unsafe_allow_html=True)


loaded_model = pickle.load(open('D://Desktop//FDM_MiniProject//FDM_MiniProject//BACKEND//logistic_regression_model.pkl', 'rb'))

#creating a function for prediction
def diabetes_prediction(input_data):  
    # Changing the input data to a numpy array
    input_data_as_numpy_array = np.asarray(input_data)

    # Reshape the array
    input_data_reshaped = input_data_as_numpy_array.reshape(1, -1)
    #print(input_data_reshaped)

    # Load the scaler
    scaler = pickle.load(open('D:\Desktop\FDM_MiniProject\FDM_MiniProject\BACKEND\scaler.sav', 'rb'))

    # Standardize the input data using the loaded scaler
    std_data = scaler.transform(input_data_reshaped)
    #print(std_data)

    # Make predictions
    prediction = loaded_model.predict(std_data)
    print(prediction)

    if prediction[0] == 0:
        return 'The person is not diabetic'
    else:
        return 'The person is diabetic'    
    
def main():

    #title
    st.title('Diabetes Prediction Web App')

    image = Image.open('D:\Desktop\FDM_MiniProject\FDM_MiniProject\FRONTEND\images\panel.jpg')
    st.image(image, use_column_width=True, clamp=False, channels="RGB", output_format="auto")
 
    
    #getting the input data from the user    
    
    Pregnancies = st.text_input('Number of Pregnancies')
    Glucose = st.text_input('Glucose Level')
    BloodPressure = st.text_input('Blood Pressure Value - (mm Hg)')
    SkinThickness = st.text_input('Skin Thickness Value - (mm)')
    Insulin = st.text_input('Insulin Level - (mu U/ml)')
    BMI = st.text_input('BMI Value - (kg/m2)')
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

 




