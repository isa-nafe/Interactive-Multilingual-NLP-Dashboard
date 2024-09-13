Interactive Multilingual NLP Insights
This project aims to provide interactive multilingual NLP (Natural Language Processing) insights using various techniques such as text analysis, sentiment analysis, topic modeling, named entity recognition, and real-time Twitter streaming. The application is built using Streamlit and offers a user-friendly interface for text analysis, classification, and real-time Twitter stream analysis.

Features
Text Analysis
Language Detection
Text Summarization
Named Entity Recognition
Sentiment Analysis
Topic Modeling
Text Classification
Real-time Twitter Stream Analysis
Installation
Ensure you have Python 3.11 or later installed. Install the required dependencies using pip:

pip install -r requirements.txt
Usage
Run the application:

streamlit run main.py
Project Structure
Directories and Files
main.py: Entry point of the application. Contains Streamlit app structure and navigation.
nlp_utils.py: Utility functions for text preprocessing, language detection, text summarization, and cross-lingual analysis.
topic_modeling.py: Functions to perform topic modeling using LDA (Latent Dirichlet Allocation).
sentiment_analysis.py: Functions to analyze sentiment using TextBlob.
visualization.py: Functions to create visualizations for sentiment and topic modeling using Plotly and Matplotlib.
text_classification.py: Functions to train and use a text classifier based on TF-IDF and Naive Bayes classifier.
twitter_stream.py: Implementation of real-time Twitter stream analysis using Tweepy and the above utilities.
Configuration
requirements.txt: Lists all the Python dependencies needed for the project.
.streamlit/config.toml: Streamlit configuration file.
.replit: Configuration file for running the project on Replit.
Example 
 Configuration
entrypoint = "main.py"
modules = ["python-3.11"]
[nix]
channel = "stable-23_05"
[unitTest]
language = "python3"
[gitHubImport]
requiredFiles = [".replit", "replit.nix"]
[deployment]
run = ["python3", "main.py"]
deploymentTarget = "cloudrun"
Detailed Explanation
main.py
Sets up the Streamlit server and page configuration.
Contains navigation for "Text Analysis", "Text Classification", and "Twitter Stream".
Uses various functions from other Python modules to perform NLP tasks and display results interactively.
nlp_utils.py
Contains core NLP utilities:

detect_language: Detects the language of the input text using langdetect.
preprocess_text: Preprocesses text by tokenizing, removing stopwords, and lemmatizing tokens.
get_named_entities: Identifies named entities in the text using SpaCy.
summarize_text: Summarizes the text using extractive summarization.
cross_lingual_analysis: Conducts cross-lingual analysis including translation to the target language.
topic_modeling.py
Implements topic modeling using Gensim's LDA model.
perform_topic_modeling handles creating the dictionary and corpus from processed text, training the model, and extracting topics.
sentiment_analysis.py
Uses TextBlob for sentiment analysis.
analyze_sentiment calculates polarity and subjectivity scores, and classifies the text as positive, neutral, or negative.
visualization.py
Creates visualizations for the NLP results:

plot_sentiment_distribution: Plots sentiment distribution using Plotly.
plot_topic_distribution: Plots topic distribution using Plotly.
generate_wordcloud: Generates a word cloud from the words using Matplotlib.
text_classification.py
Implements text classification using TF-IDF Vectorizer and Naive Bayes classifier.
train_text_classifier trains the text classifier.
classify_text classifies new texts based on the trained model.
twitter_stream.py
Implements a real-time Twitter stream using Tweepy.
run_twitter_stream: Starts the Twitter stream for real-time analysis.
Processes each tweet for language detection, sentiment analysis, and topic modeling.
Running the Application
Text Analysis:

Enter text or upload a text file.
Option to select target language for cross-lingual analysis.
Displays analysis results including language detection, text summarization, named entity recognition, sentiment analysis, and topic modeling.
Text Classification:

Input training data and train a text classifier.
Input text to classify and view the prediction and class probabilities.
Twitter Stream:

Enter keywords to track on Twitter.
Start real-time Twitter stream to view analysis of incoming tweets.