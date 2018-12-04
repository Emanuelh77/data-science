import nltk
import numpy as np
import random
import string
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


f = open('chatbot.txt','r',errors = 'ignore')

raw = f.read()
raw = raw.lower()

nltk.download('punkt')
nltk.download('wordnet')

sent_tokens = nltk.sent_tokenize(raw)       #converts to a list of sentences
word_tokens = nltk.word_tokenize(raw)       #converts to a list of words

print(sent_tokens[:2])

lemmer = nltk.stem.WordNetLemmatizer() #wordnet is a semantically-oriented English dict in NLTK

#Pre-processing raw data

def lem_tokens(tokens):
    return [lemmer.lemmatize(token) for token in tokens]
remove_punct_dict = dict((ord(punct), None) for punct in string.punctuation)

def lem_normalize(text):
    return lem_tokens(nltk.word_tokenize(text.lower().translate(remove_punct_dict)))

#Keyword matching

GREETING_INPUTS = ("hello", "hi", "greetings", "sup", "what's up", "hey", "how you doing?")
GREETING_RESPONSES = ["hi", "hey", "*nods*", "hi there", "hello", "I am glad! You are talking to me"]
def greeting(sentence):
    for word in sentence.split():
        if word.lower() in GREETING_INPUTS:
            return random.choice(GREETING_RESPONSES)

#Generating a response

