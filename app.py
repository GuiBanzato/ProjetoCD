import streamlit as st
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import tensorflow_hub as hub
import joblib
import re
import numpy as np

# Load the trained model
clf = joblib.load('modelo_treinado.pkl')

# Load the Universal Sentence Encoder
embed = hub.load("https://tfhub.dev/google/universal-sentence-encoder/4")

# Function for prediction
def predict_category(text, model, embedder):
    # Clean the text (same cleaning as during training)
    text = re.sub(r'\s+', ' ', text)  
    text = re.sub(r'\W', ' ', text)   
    text = text.lower()               

    # Get the embedding
    embedding = embedder([text])[0].numpy()

    # Make the prediction
    prediction = model.predict_proba([embedding])  # Note the [embedding]
    return prediction

# Streamlit configuration
st.title("ArXiv Predição de Categoria")
input_text = st.text_area("Enter article summary:", "")

if st.button("Predict"):
    if input_text:
        pred = predict_category(input_text, clf, embed)
        st.write("Probabilidade de Classe:")
        
        # Assuming you have two categories: cs.AI and stat.ML
        categories = ['cs.AI', 'stat.ML', 'cs.LG', 'cs.CR' ]  
        
        # Create a DataFrame for the bar chart
        pred_df = pd.DataFrame(pred, columns=categories)  
        st.bar_chart(pred_df) 
    else:
        st.warning("Escreva algo.")
