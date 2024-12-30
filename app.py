import streamlit as st
import joblib
import json
import tensorflow as tf
from tensorflow.keras.models import load_model, save_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer
import numpy as np

# Load the trained model and tokenizer
@st.cache_resource
def load_resources():
    try:
        model = tf.keras.models.load_model('saved_model_format')
        tokenizer = joblib.load('tokenizer.joblib')
        print(tokenizer)
        return model, tokenizer
    except Exception as e:
        print(f"Error loading model: {e}")
        raise


model, tokenizer = load_resources()

# Define constants
MAX_SEQUENCE_LENGTH = 30

# Title
st.title("Sarcasm Detection Model")

# Input form
input_text = st.text_input("Enter a sentence to check for sarcasm:", "")

if st.button("Predict"):
    if input_text.strip():
        # Preprocess the input text
        sequences = tokenizer.texts_to_sequences([input_text])
        padded_seq = pad_sequences(sequences, maxlen=MAX_SEQUENCE_LENGTH)
        
        # Predict using the model
        prediction = model.predict(padded_seq)[0][0]
        
        
        # Output prediction
        st.subheader("Prediction:")
        if prediction > 0.5:
            st.write(f"**Sarcastic** with a probability of {prediction:.2f}")
        else:
            st.write(f"**Not Sarcastic** with a probability of {1 - prediction:.2f}")
    else:
        st.warning("Please enter a valid sentence.")

st.markdown("This app uses an LSTM model trained on sarcasm detection data.")
