import tweepy
import streamlit as st
from nlp_utils import preprocess_text, detect_language, cross_lingual_analysis
from sentiment_analysis import analyze_sentiment
from topic_modeling import perform_topic_modeling

class TwitterStreamListener(tweepy.StreamingClient):
    def __init__(self, bearer_token, queue):
        super().__init__(bearer_token)
        self.queue = queue

    def on_tweet(self, tweet):
        if tweet.data['text'] is not None:
            self.queue.put(tweet.data['text'])

    def on_error(self, status):
        if status == 420:
            return False

def start_twitter_stream(bearer_token, queue, keywords):
    stream = TwitterStreamListener(bearer_token, queue)
    stream.add_rules(tweepy.StreamRule(" OR ".join(keywords)))
    stream.filter(tweet_fields=["created_at"])

def process_tweet(tweet):
    # Detect language
    lang = detect_language(tweet)

    # Preprocess the tweet
    processed_text, _ = preprocess_text(tweet)

    # Perform sentiment analysis
    sentiment = analyze_sentiment(processed_text)

    # Perform topic modeling
    topics, _ = perform_topic_modeling(processed_text)

    return {
        "text": tweet,
        "language": lang,
        "sentiment": sentiment,
        "topics": topics
    }

# Note: In a real application, you would need to securely manage these credentials
# and not hardcode them in the script.
TWITTER_BEARER_TOKEN = "YOUR_TWITTER_BEARER_TOKEN"

def run_twitter_stream():
    st.title("Real-time Twitter Stream Analysis")

    # Input for keywords to track
    keywords = st.text_input("Enter keywords to track (comma-separated):", "python,data science,AI")
    keywords = [k.strip() for k in keywords.split(",")]

    if st.button("Start Streaming"):
        queue = Queue()
        stream_thread = threading.Thread(target=start_twitter_stream, args=(TWITTER_BEARER_TOKEN, queue, keywords))
        stream_thread.start()

        tweet_container = st.empty()
        while True:
            if not queue.empty():
                tweet = queue.get()
                processed_tweet = process_tweet(tweet)
                
                tweet_container.write(f"""
                Tweet: {processed_tweet['text']}
                Language: {processed_tweet['language']}
                Sentiment: {processed_tweet['sentiment']['sentiment']}
                Topics: {', '.join([t[1] for t in processed_tweet['topics'][:3]])}
                """)
            time.sleep(1)

if __name__ == "__main__":
    run_twitter_stream()
