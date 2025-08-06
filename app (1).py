import streamlit as st
import pickle
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Load model and tokenizer
model = load_model('disaster_tweet_model.h5')
tokenizer = pickle.load(open('tokenizer.pkl', 'rb'))

st.title(" Disaster Tweet Classifier")

tweet = st.text_area("Enter a tweet")

if st.button("Classify"):
    seq = tokenizer.texts_to_sequences([tweet])
    padded = pad_sequences(seq, maxlen=100)
    prediction = model.predict(padded)[0][0]
    label = " Disaster Tweet" if prediction >= 0.5 else " Not a Disaster Tweet"
    st.markdown(f"**Prediction:** {label}")
