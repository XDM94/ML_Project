import streamlit as st
from transformers import pipeline


model = pipeline(task="text-classification", model="SamLowe/roberta-base-go_emotions", top_k=None)

@st.cache(allow_output_mutation=True)
def root():
    def load_model():
        return pipeline

st.title('Emotion Analysis')

text = st.text_input('Enter your emotions')

submit = st.button('Predict')

if submit:
    prediction = model.predict([text])
    st.write(prediction[0])

    print(prediction[0])
