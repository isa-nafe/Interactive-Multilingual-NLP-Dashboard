import plotly.graph_objs as go
from wordcloud import WordCloud
import matplotlib.pyplot as plt

def plot_sentiment_distribution(sentiment_scores):
    labels = ['Positive', 'Neutral', 'Negative']
    values = [
        max(0, sentiment_scores['polarity']),
        1 - abs(sentiment_scores['polarity']),
        max(0, -sentiment_scores['polarity'])
    ]

    fig = go.Figure(data=[go.Pie(labels=labels, values=values, hole=.3)])
    fig.update_layout(title_text="Sentiment Distribution")
    return fig

def plot_topic_distribution(topic_distribution):
    topics = [f"Topic {i+1}" for i, _ in topic_distribution]
    relevance = [score for _, score in topic_distribution]

    fig = go.Figure(data=[go.Bar(x=topics, y=relevance)])
    fig.update_layout(title_text="Topic Distribution", xaxis_title="Topics", yaxis_title="Relevance")
    return fig

def generate_wordcloud(words):
    wordcloud = WordCloud(width=800, height=400, background_color='white').generate(" ".join(words))
    
    plt.figure(figsize=(10, 5))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    plt.tight_layout(pad=0)
    
    return wordcloud
