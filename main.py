# Step 1: Import Libraries and Load the Model

import numpy as np
import tensorflow as tf
import streamlit as st

from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing import sequence
from tensorflow.keras.models import load_model


# Load IMDB dataset word index
word_index = imdb.get_word_index()

# Create reverse word index for decoding
reverse_word_index = {value: key for key, value in word_index.items()}


# Load the pre-trained model
# compile=False avoids compatibility issues with older .h5 models
model = load_model("simple_rnn_imdb.h5", compile=False)


# Step 2: Helper Functions

# Function to decode reviews
def decode_review(encoded_review):
    return " ".join(
        [reverse_word_index.get(i - 3, "?") for i in encoded_review]
    )


# Function to preprocess user input
def preprocess_text(text):
    words = text.lower().split()

    # Convert words to IMDB integer sequence
    encoded_review = [
        word_index.get(word, 2) + 3 for word in words
    ]

    # Pad sequence to model input length
    padded_review = sequence.pad_sequences(
        [encoded_review],
        maxlen=500
    )

    return padded_review


# Step 3: Streamlit App

st.title("IMDB Movie Review Sentiment Analysis")

st.write(
    "Enter a movie review to classify it as Positive or Negative."
)


# User input
user_input = st.text_area("Movie Review")


# Prediction button
if st.button("Classify"):

    if user_input.strip() == "":
        st.warning("Please enter a movie review.")

    else:
        # Preprocess input
        preprocessed_input = preprocess_text(user_input)

        # Make prediction
        prediction = model.predict(preprocessed_input)

        score = prediction[0][0]

        sentiment = "Positive" if score > 0.5 else "Negative"

        # Display result
        st.success(f"Sentiment: {sentiment}")
        st.write(f"Prediction Score: {score:.4f}")

else:
    st.info("Please enter a movie review.")