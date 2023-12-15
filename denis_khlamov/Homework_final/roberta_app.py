import streamlit as st
from transformers import pipeline
import time


model = pipeline(task="text-classification", model="SamLowe/roberta-base-go_emotions", top_k=None)

st.title('Emotion Analysis')

text = st.text_input('Enter your emotions')

submit = st.button('Predict')

if submit:
    prediction = model.predict([text])
    st.write(prediction[0])

    print(prediction[0])

if submit:
    start = time.time()
    prediction = model.predict([text])
    end = time.time()
    st.write('Prediction time taken: ', round(end - start, 2), 'seconds')

    print(prediction[0])
    st.write(prediction[0])