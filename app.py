import pickle  # Standard library

import streamlit as st  # Third-party libraries
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Cache the model and tokenizer loading for better performance
@st.cache_resource
def load_sentiment_model():
    try:
        return load_model('sentiment_lstm_model.h5')
    except Exception as e:
        st.error(f"❌ Error loading model: {e}")
        return None

@st.cache_resource
def load_tokenizer():
    try:
        with open('tokenizer.pkl', 'rb') as file:
            return pickle.load(file)
    except Exception as e:
        st.error(f"❌ Error loading tokenizer: {e}")
        return None

# Load model and tokenizer once
model = load_sentiment_model()
tokenizer = load_tokenizer()

# Streamlit app UI
st.title("🎬 IMDB Movie Review Sentiment Analysis")

# User input for movie review
review = st.text_area("Enter a movie review:", placeholder="Type your review here...")

if st.button("Analyze Sentiment"):
    if not review.strip():
        st.error("❗ Please enter a valid movie review.")
    else:
        if model is None or tokenizer is None:
            st.error("❌ Model or tokenizer not loaded. Please check the files.")
        else:
            # Preprocess the input review
            sequence = tokenizer.texts_to_sequences([review.strip()])
            padded_sequence = pad_sequences(sequence, maxlen=200)

            # Predict sentiment
            prediction = model.predict(padded_sequence)
            sentiment = "😃 Positive" if prediction[0][0] > 0.5 else "☹️ Negative"

            # Display the sentiment result
            st.markdown("### Sentiment Analysis Result:")
            st.success(sentiment)
