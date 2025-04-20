#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr 20 11:22:18 2025

@author: peerachet
"""
import streamlis as st
import numpy as np
import pickle

#load model
with open ('dtm_trained_model.plk','rb') as f:
    dtm_model = pickle.load(f)
    
st.title("Iris flower Classification")
st.write("Enter the feature of the iris flower: ")

sepal_length = st.slider("Sepal Length (cm)",4.0,8.0,5.1)
sepal_width = st.slider("Sepal_width (cm)",2.0,4.5,3.5)
petal_length = st.slider("Petal_length (cm)",1.0,7.0,1.4)
petal_width= st.slider("Petal_width (cm)",0.1,2.5,0.2)

if st.button("Predict"):
    input_data = np.array([[sepal_length],[sepal_width],[petal_length],[petal_width]])
    prediction = dtm_model.predict(input_data)
    speicies = ['Setosa','Versicolor','Virgigica']
    st.success (f"The predicted species is: **{species[prediction[0]]}**")
