from textblob import TextBlob

def analyze_sentiment(processed_text):
    # Join the processed tokens back into a string
    text = " ".join(processed_text)

    # Perform sentiment analysis using TextBlob
    blob = TextBlob(text)
    polarity = blob.sentiment.polarity
    subjectivity = blob.sentiment.subjectivity

    # Determine overall sentiment
    if polarity > 0:
        sentiment = "Positive"
    elif polarity < 0:
        sentiment = "Negative"
    else:
        sentiment = "Neutral"

    return {
        "sentiment": sentiment,
        "polarity": polarity,
        "subjectivity": subjectivity
    }
