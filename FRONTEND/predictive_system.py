# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import numpy as np
import pickle
from sklearn.preprocessing import StandardScaler

# Loading the saved model
loaded_model = pickle.load(open('E://3rd year//Y3S2//FDM//Mini Project//trained_model.sav','rb'))

input_data = (6,148,72,35,0,33.6,0.627,50)
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
    print('The person is not diabetic')
else:
    print('The person is diabetic')


