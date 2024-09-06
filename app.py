import pandas as pd
import pickle as pk
from sklearn.feature_extraction.text import TfidfVectorizer
import streamlit as st

# Load the pre-trained model and scaler
model = pk.load(open('model.pkl', 'rb'))
scaler = pk.load(open('scaler.pkl', 'rb'))

# Set the title and favicon of the application
st.set_page_config(page_title='Analysis', page_icon='🎥', layout='centered')

# Create a custom header with a catchy title and icon
st.markdown("<h1 style='text-align: center; color: white;'>🎬 Movie Review Analysis 🍿</h1>", unsafe_allow_html=True)
st.markdown("<h3 style='text-align: center; color: lightgray;'>Discover the sentiment behind your favorite movies!</h3>", unsafe_allow_html=True)

# Create a text area input for the user to enter a movie review
review = st.text_area('Enter Movie Review', height=150, placeholder="Type your movie review here...")

# Create a button for the user to trigger the prediction with a custom style
if st.button('Predict Sentiment', help="Click to analyze the sentiment"):
    if review.strip():
        # Transform the input review using the scaler
        review_scale = scaler.transform([review]).toarray()
        
        # Use the pre-trained model to predict the sentiment
        result = model.predict(review_scale)
        
        # Display the predicted sentiment with emojis and custom styling
        if result[0] == 0:
            st.markdown("<h2 style='text-align: center; color: red;'>😞 Negative Review 😞</h2>", unsafe_allow_html=True)
        else:
            st.markdown("<h2 style='text-align: center; color: green;'>😊 Positive Review 😊</h2>", unsafe_allow_html=True)
    else:
        st.warning("Please enter a review before clicking 'Predict Sentiment'.")
else:
    st.info("Type a movie review and click 'Predict Sentiment' to see the result.")
