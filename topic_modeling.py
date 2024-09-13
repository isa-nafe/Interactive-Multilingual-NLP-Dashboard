from gensim import corpora
from gensim.models import LdaModel
from gensim.models.coherencemodel import CoherenceModel

def perform_topic_modeling(processed_text, num_topics=5):
    # Create a dictionary from the processed text
    dictionary = corpora.Dictionary([processed_text])

    # Create a corpus
    corpus = [dictionary.doc2bow(text) for text in [processed_text]]

    # Train the LDA model
    lda_model = LdaModel(corpus=corpus, id2word=dictionary, num_topics=num_topics, random_state=42)

    # Get the topics
    topics = lda_model.print_topics(num_words=10)

    # Get the topic distribution for the input text
    topic_distribution = lda_model.get_document_topics(corpus[0])

    # Sort topics by relevance
    sorted_topics = sorted(topic_distribution, key=lambda x: x[1], reverse=True)

    return topics, sorted_topics
