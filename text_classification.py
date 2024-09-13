from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import numpy as np

def train_text_classifier(texts, labels):
    # Convert text to TF-IDF features
    vectorizer = TfidfVectorizer(max_features=5000)
    X = vectorizer.fit_transform(texts)
    
    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, labels, test_size=0.2, random_state=42)
    
    # Train a Naive Bayes classifier
    clf = MultinomialNB()
    clf.fit(X_train, y_train)
    
    # Evaluate the model
    y_pred = clf.predict(X_test)
    report = classification_report(y_test, y_pred)
    
    return vectorizer, clf, report

def classify_text(vectorizer, clf, text):
    # Transform the input text
    X = vectorizer.transform([text])
    
    # Predict the class
    prediction = clf.predict(X)[0]
    
    # Get probability scores for each class
    proba = clf.predict_proba(X)[0]
    
    return prediction, proba

# Example usage:
# texts = ["This is a positive review", "I hate this product", "The movie was okay"]
# labels = ["positive", "negative", "neutral"]
# vectorizer, clf, report = train_text_classifier(texts, labels)
# prediction, proba = classify_text(vectorizer, clf, "I love this product")
