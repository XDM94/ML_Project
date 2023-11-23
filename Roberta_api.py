from fastapi import FastAPI
from transformers import pipeline

app = FastAPI()
classifier = pipeline(task="text-classification", model="SamLowe/roberta-base-go_emotions", top_k=None)

@app.get("/")
def root():
    return {"message": "Hello World"}

@app.get("/predict/")
def predict():
    return classifier("I am not having a great day")