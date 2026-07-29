import numpy as np
import tensorflow as tf
import streamlit as st

from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing import sequence
from tensorflow.keras.models import load_model


# -----------------------------
# Load IMDB word index
# -----------------------------

word_index = imdb.get_word_index()

reverse_word_index = {
    value: key for key, value in word_index.items()
}


# -----------------------------
# Load Model
# -----------------------------

try:
    model = load_model(
        "simple_rnn_imdb.h5",
        compile=False
    )

except Exception as e:
    st.error("Model loading failed:")
    st.write(e)
    st.stop()


# -----------------------------
# Helper Functions
# -----------------------------

def decode_review(encoded_review):
    decoded = []

    for i in encoded_review:
        decoded.append(
            reverse_word_index.get(i - 3, "?")
        )

    return " ".join(decoded)



def preprocess_text(text):

    words = text.lower().split()

    encoded_review = []

    for word in words:
        encoded_review.append(
            word_index.get(word, 2) + 3
        )

    padded_review = sequence.pad_sequences(
        [encoded_review],
        maxlen=500
    )

    return padded_review



# -----------------------------
# Streamlit App
# -----------------------------

st.title("IMDB Movie Review Sentiment Analysis")

st.write(
    "Enter a movie review and the model will predict Positive or Negative sentiment."
)


user_input = st.text_area(
    "Movie Review"
)


if st.button("Classify"):

    if user_input.strip() == "":
        st.warning("Please enter a movie review.")

    else:

        processed_input = preprocess_text(
            user_input
        )

        prediction = model.predict(
            processed_input
        )

        score = float(prediction[0][0])

        if score > 0.5:
            sentiment = "Positive"
        else:
            sentiment = "Negative"


        st.success(
            f"Sentiment: {sentiment}"
        )

        st.write(
            f"Prediction Score: {score:.4f}"
        )


else:
    st.info(
        "Please enter a movie review."
    )