
import streamlit as st
import pickle
import matplotlib.pyplot as plt
from textblob import TextBlob
import pandas as pd

# Load model
with open('sentiment_model.pkl', 'rb') as f:
    model = pickle.load(f)
with open('tfidf_vectorizer.pkl', 'rb') as f:
    tfidf = pickle.load(f)

# Page config
st.set_page_config(page_title="Travel Sentiment Analyzer", page_icon="✈️", layout="wide")

st.title("✈️ Travel Review Sentiment Analyzer")
st.markdown("### AI-powered hotel review analysis")

# Input
review = st.text_area("Enter your hotel review:", height=150, 
                       placeholder="e.g. The hotel was amazing, great service and clean rooms!")

if st.button("Analyze Review 🔍"):
    if review:
        # Sentiment
        blob = TextBlob(review)
        sentiment_score = blob.sentiment.polarity
        
        if sentiment_score > 0:
            sentiment = "😊 Positive"
            color = "green"
        elif sentiment_score < 0:
            sentiment = "😞 Negative"  
            color = "red"
        else:
            sentiment = "😐 Neutral"
            color = "gray"
        
        # Rating prediction
        review_tfidf = tfidf.transform([review])
        predicted_rating = model.predict(review_tfidf)[0]
        
        # Display results
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Sentiment", sentiment)
        with col2:
            st.metric("Sentiment Score", f"{sentiment_score:.2f}")
        with col3:
            st.metric("Predicted Rating", f"⭐ {predicted_rating}/5")
            
        st.success("Analysis Complete!")
    else:
        st.warning("Please enter a review first!")
