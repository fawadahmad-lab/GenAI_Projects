import streamlit as st
from transformers import pipeline

# Load pre-trained sentiment analysis pipeline from Hugging Face
sentiment_pipeline = pipeline("sentiment-analysis" , model='distilbert/distilbert-base-uncased-finetuned-sst-2-english')

# Custom CSS for a minimalist and attractive UI
st.markdown("""
    <style>
        /* Set background color */
        .reportview-container {
            background-color: #f4f4f9;
        }

        /* Title color */
        .title {
            color: #4a90e2;
            font-size: 40px;
            font-weight: bold;
        }

        /* Input field background */
        .stTextInput textarea {
            background-color: #ffffff;
            color: #333;
            border: 2px solid #4a90e2;
            border-radius: 5px;
            padding: 10px;
            font-size: 16px;
        }

        /* Button Style */
        .stButton>button {
            background-color: #4a90e2;
            color: white;
            border-radius: 5px;
            padding: 10px 20px;
            border: none;
            font-size: 16px;
        }

        /* Set the results style */
        .result-text {
            font-size: 18px;
            font-weight: bold;
            color: #4a90e2;
        }

        .result-score {
            font-size: 16px;
            color: #666;
        }

        /* Center everything */
        .main {
            text-align: center;
        }
    </style>
""", unsafe_allow_html=True)

# Set up Streamlit interface
st.markdown('<p class="title">Sentiment Analysis GenAI</p>', unsafe_allow_html=True)
st.write("This app performs sentiment analysis on text input. Enter some text below to analyze its sentiment.")

# Text input box for user to type their sentence
user_input = st.text_area("Enter text for sentiment analysis:", height=150)

# Button to analyze sentiment
if st.button("Analyze Sentiment"):
    if user_input:
        # Get the sentiment of the input text
        result = sentiment_pipeline(user_input)
        
        # Display the result
        sentiment = result[0]['label']
        score = result[0]['score']
        
        st.markdown(f"<p class='result-text'>Sentiment: {sentiment}</p>", unsafe_allow_html=True)
        st.markdown(f"<p class='result-score'>Confidence: {score:.2f}</p>", unsafe_allow_html=True)
    else:
        st.write("Please enter some text to analyze.")
