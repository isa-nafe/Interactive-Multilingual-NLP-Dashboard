import spacy
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from langdetect import detect
from textblob import TextBlob

# Download necessary NLTK data
nltk.download('punkt')
nltk.download('stopwords')

# Load spaCy models for different languages
nlp_models = {
    'en': spacy.load("en_core_web_sm"),
    'es': spacy.load("es_core_news_sm"),
    'fr': spacy.load("fr_core_news_sm"),
    'de': spacy.load("de_core_news_sm")
}

def detect_language(text):
    try:
        return detect(text)
    except:
        return 'en'  # Default to English if detection fails

def preprocess_text(text):
    lang = detect_language(text)
    if lang not in nlp_models:
        lang = 'en'  # Default to English if the detected language is not supported
    
    nlp = nlp_models[lang]
    doc = nlp(text)
    
    # Tokenize and remove stopwords
    tokens = [token.text.lower() for token in doc if not token.is_stop and token.is_alpha]
    
    # Lemmatize tokens
    lemmatized_tokens = [token.lemma_ for token in nlp(" ".join(tokens))]
    
    return lemmatized_tokens, lang

def get_named_entities(text):
    lang = detect_language(text)
    if lang not in nlp_models:
        lang = 'en'  # Default to English if the detected language is not supported
    
    nlp = nlp_models[lang]
    doc = nlp(text)
    entities = [(ent.text, ent.label_) for ent in doc.ents]
    return entities, lang

def summarize_text(text, num_sentences=3):
    lang = detect_language(text)
    if lang not in nlp_models:
        lang = 'en'  # Default to English if the detected language is not supported
    
    nlp = nlp_models[lang]
    doc = nlp(text)
    sentences = [sent.text.strip() for sent in doc.sents]
    
    # Simple extractive summarization using the first few sentences
    summary = " ".join(sentences[:num_sentences])
    return summary, lang

def translate_text(text, target_lang='en'):
    source_lang = detect_language(text)
    if source_lang == target_lang:
        return text
    
    blob = TextBlob(text)
    translated = blob.translate(to=target_lang)
    return str(translated)

def cross_lingual_analysis(text, target_lang='en'):
    original_lang = detect_language(text)
    
    if original_lang != target_lang:
        translated_text = translate_text(text, target_lang)
    else:
        translated_text = text
    
    processed_text, _ = preprocess_text(translated_text)
    entities, _ = get_named_entities(translated_text)
    summary, _ = summarize_text(translated_text)
    
    return {
        'original_language': original_lang,
        'target_language': target_lang,
        'processed_text': processed_text,
        'entities': entities,
        'summary': summary
    }
