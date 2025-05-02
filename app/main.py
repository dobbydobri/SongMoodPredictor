import streamlit as st

st.title("🎵 Mood Predictor")

st.write("This app will predict the mood of a song based on audio features.")

# Example dropdown (to be replaced with real song list later)
song = st.selectbox("Choose a song:", ["Song A", "Song B", "Song C"])

if st.button("Predict Mood"):
    # This will be replaced with your actual ML model
    st.success(f"The predicted mood for '{song}' is: Happy 😊")
