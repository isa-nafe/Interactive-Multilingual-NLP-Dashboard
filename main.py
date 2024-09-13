import streamlit as st
import pandas as pd
import nltk
from nlp_utils import preprocess_text, get_named_entities, summarize_text, detect_language, cross_lingual_analysis
from topic_modeling import perform_topic_modeling
from sentiment_analysis import analyze_sentiment
from visualization import plot_sentiment_distribution, plot_topic_distribution, generate_wordcloud
from text_classification import train_text_classifier, classify_text
from twitter_stream import run_twitter_stream

# Download necessary NLTK data
nltk.download('punkt')
nltk.download('stopwords')

st.set_page_config(page_title="Multilingual NLP Insights", layout="wide")

st.title("Interactive Multilingual NLP Insights")

# Sidebar for navigation
page = st.sidebar.selectbox("Choose a page", ["Text Analysis", "Text Classification", "Twitter Stream"])

if page == "Text Analysis":
    # Input text area
    text_input = st.text_area("Enter your text here:", height=200)

    # File upload option
    uploaded_file = st.file_uploader("Or upload a text file:", type=["txt"])

    if uploaded_file is not None:
        text_input = uploaded_file.getvalue().decode("utf-8")

    if text_input:
        # Detect language
        detected_lang = detect_language(text_input)
        st.write(f"Detected language: {detected_lang}")

        # Language selection for analysis
        target_lang = st.selectbox("Select target language for analysis:", ["en", "es", "fr", "de"])

        # Perform cross-lingual analysis
        analysis_results = cross_lingual_analysis(text_input, target_lang)

        # Display results
        st.header("Analysis Results")

        # Original and translated text
        st.subheader("Original Text")
        st.write(text_input)
        if detected_lang != target_lang:
            st.subheader(f"Translated Text ({target_lang})")
            st.write(analysis_results['summary'])

        # Text Summary
        st.subheader("Text Summary")
        st.write(analysis_results['summary'])

        # Named Entity Recognition
        st.subheader("Named Entities")
        entity_df = pd.DataFrame(analysis_results['entities'], columns=["Entity", "Type"])
        st.dataframe(entity_df)

        # Sentiment Analysis
        st.subheader("Sentiment Analysis")
        sentiment_scores = analyze_sentiment(analysis_results['processed_text'])
        col1, col2 = st.columns(2)
        with col1:
            st.write(f"Overall Sentiment: {sentiment_scores['sentiment']}")
            st.write(f"Polarity: {sentiment_scores['polarity']:.2f}")
            st.write(f"Subjectivity: {sentiment_scores['subjectivity']:.2f}")
        with col2:
            fig_sentiment = plot_sentiment_distribution(sentiment_scores)
            st.plotly_chart(fig_sentiment)

        # Topic Modeling
        st.subheader("Topic Modeling")
        topics, topic_distribution = perform_topic_modeling(analysis_results['processed_text'])
        fig_topics = plot_topic_distribution(topic_distribution)
        st.plotly_chart(fig_topics)

        # Display topics and word clouds
        for i, (topic, words) in enumerate(topics):
            st.write(f"Topic {i+1}: {', '.join(words[:10])}")
            wordcloud = generate_wordcloud(words)
            st.image(wordcloud.to_array())

elif page == "Text Classification":
    st.header("Text Classification")
    st.write("To use text classification, you need to provide training data.")
    
    # Input for training data
    train_data = st.text_area("Enter training data (one example per line, format: text|||label):", height=100)
    
    if train_data:
        # Process training data
        train_examples = [line.split("|||") for line in train_data.split("\n") if "|||" in line]
        texts, labels = zip(*train_examples)
        
        # Train the classifier
        vectorizer, clf, report = train_text_classifier(texts, labels)
        
        st.write("Classification Report:")
        st.code(report)
        
        # Input for text to classify
        text_to_classify = st.text_area("Enter text to classify:", height=100)
        
        if text_to_classify:
            # Classify the input text
            prediction, proba = classify_text(vectorizer, clf, text_to_classify)
            
            st.write(f"Predicted class for input text: {prediction}")
            st.write("Class probabilities:")
            for label, prob in zip(clf.classes_, proba):
                st.write(f"{label}: {prob:.2f}")

elif page == "Twitter Stream":
    run_twitter_stream()

# Add some information about the app
st.sidebar.header("About")
st.sidebar.info("This application performs multilingual sentiment analysis, topic modeling, named entity recognition, text summarization, and text classification on input text. It also includes real-time processing of Twitter data. It provides visual insights into the sentiment distribution, main topics, and entities present in the text.")

# Add instructions
st.sidebar.header("Instructions")
st.sidebar.markdown("""
1. Choose a page from the dropdown menu.
2. For Text Analysis:
   - Enter text in the text area or upload a text file.
   - The app will detect the language of the input text.
   - Select the target language for analysis.
   - Explore the text summary, named entities, sentiment analysis, and topic modeling visualizations.
3. For Text Classification:
   - Provide training data in the specified format.
   - Enter text to classify and see the results.
4. For Twitter Stream:
   - Enter keywords to track.
   - Start the stream to see real-time analysis of tweets.
""")
