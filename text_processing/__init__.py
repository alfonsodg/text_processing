#!/usr/bin/python2
# -*- coding: utf-8 -*-

import re
import pickle
import string
import vincent
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
#from sklearn.feature_extraction.text import TfidfVectorizer
from nltk import bigrams, trigrams, FreqDist
from nltk.stem import SnowballStemmer
from nltk.corpus import stopwords
from nltk import wordpunct_tokenize
from gensim.models import Word2Vec
from sklearn.externals import joblib


snowball_stemmer = SnowballStemmer("spanish")

with open('/opt/machine_learner_data/tendencias_trigram_tagger_ia.dat', 'r') as fd:
    tagger = pickle.load(fd)

model = Word2Vec.load('/opt/machine_learner_data/tendencias_word2vec_ia.dat')

#sentiment_headers = pickle.load(open('../machine_learner_data/tendencias_sentiment_ia_headers.dat', 'rb'))
sentiment_vectorizer = joblib.load('/opt/machine_learner_data/tendencias_sentiment_vectorizer.dat')
sentiment_classifier = joblib.load('/opt/machine_learner_data/tendencias_sentiment_ia.dat')

topic_headers = pickle.load(open('/opt/machine_learner_data/tendencias_topics_ia_headers.dat', 'rb'))
topic_classifier = joblib.load('/opt/machine_learner_data/tendencias_topics_ia.dat')


emoticons_str = r"""
    (?:
        [:=;] # Eyes
        [oO\-]? # Nose (optional)
        [D\)\]\(\]/\\OpP] # Mouth
    )"""

regex_str = [
    emoticons_str,
    r'<[^>]+>', # HTML tags
    r'(?:@[\w_]+)', # @-mentions
    r"(?:\#+[\w_]+[\w\'_\-]*[\w_]+)", # hash-tags
    r'http[s]?://(?:[a-z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-f][0-9a-f]))+', # URLs
    r'(?:(?:\d+,?)+(?:\.?\d+)?)', # numbers
    #    r"(?:[a-z][a-z'\-_]+[a-z])", # words with - and '
    r'(?:[\w_]+)', # other words
    r'(?:\S)' # anything else
]

extra_terms = [
    'rt',
    'via',
    'RT',
    'VIA'
]

extra_chars = [
    u'”',
    u'“',
    u'‘',
    u'¡',
    u'¿',
    u'’',
    u'►',
    u'…',
    u'..',
    u'...',
]

punctuation = list(string.punctuation)
stop = stopwords.words('spanish') + punctuation + extra_terms + extra_chars
tokens_re = re.compile(r'(' + '|'.join(regex_str) + ')', re.VERBOSE | re.IGNORECASE | re.UNICODE)
emoticon_re = re.compile(r'^' + emoticons_str + '$', re.VERBOSE | re.IGNORECASE | re.UNICODE)


def _calculate_languages_ratios(text):
    """
    Calculate probability of given text to be written in several languages and
    return a dictionary that looks like {'french': 2, 'spanish': 4, 'english': 0}

    @param text: Text whose language want to be detected
    @type text: str

    @return: Dictionary with languages and unique stopwords seen in analyzed text
    @rtype: dict
    """
    languages_ratios = {}
    tokens = wordpunct_tokenize(text)
    words = [word.lower() for word in tokens]
    for language in stopwords.fileids():
        stopwords_set = set(stopwords.words(language))
        words_set = set(words)
        common_elements = words_set.intersection(stopwords_set)
        languages_ratios[language] = len(common_elements) # language "score"
    return languages_ratios


def detect_language(text):
    """
    Calculate probability of given text to be written in several languages and
    return the highest scored.

    It uses a stopwords based approach, counting how many unique stopwords
    are seen in analyzed text.

    @param text: Text whose language want to be detected
    @type text: str

    @return: Most scored language guessed
    @rtype: str
    """
    ratios = _calculate_languages_ratios(text)
    most_rated_language = max(ratios, key=ratios.get)
    return most_rated_language


def list_to_dict(data_list):
    cols = dict((key, val) for key, val in enumerate(data_list))
    return cols


# def stem_tokens(tokens, stemmer):
#     stemmed = []
#     for item in tokens:
#         stemmed.append(stemmer.stem(item))
#     return stemmed
#
# def tokenize_stem(text):
#     tokens = nltk.word_tokenize(text)
#     stems = stem_tokens(tokens, snowball_stemmer)
#     return stems

def stemming(word):
    stemmed = snowball_stemmer.stem(word)
    return stemmed


def tokenize(sentence):
    return tokens_re.findall(sentence)


def pre_process(text, lowercase=True):
    tokens = tokenize(text)
    if lowercase:
        tokens = [token if emoticon_re.search(token) else token.lower() for token in tokens]
    return tokens


def terms_builder(content, stop_words=stop, stem=False):
    full_terms = []
    sentence_terms = []
    for text in content:
        tmp_terms = []
        for term in pre_process(text, lowercase=True):
            if term not in stop_words:
                if stem:
                    term = stemming(term)
                full_terms.append(term)
                tmp_terms.append(term)
        sentence_terms.append(tmp_terms)
    return full_terms, sentence_terms


def uni_gram_finder(content):
    terms = [term for text in content for term in pre_process(text, lowercase=True) if term not in stop]
    return terms


def bi_gram_finder(tokens):
    return [(tokens[i], tokens[i + 1]) for i in range(len(tokens) - 1)]


def tri_gram_finder(tokens):
    return [(tokens[i], tokens[i + 1], tokens[i + 2]) for i in range(len(tokens) - 2)]


def n_gram_nltk(terms):
    terms_bi_gram = bigrams(terms)
    terms_tri_gram = trigrams(terms)
    uni_gram_matrix = FreqDist(terms)
    bi_gram_matrix = FreqDist(terms_bi_gram)
    tri_gram_matrix = FreqDist(terms_tri_gram)
    return uni_gram_matrix.items(), bi_gram_matrix.items(), tri_gram_matrix.items()


def n_gram_counter(content, ngram_range=(1,3), stop_words=stop):
    vectorizer = CountVectorizer(
        tokenizer=pre_process,
        lowercase=False,
        stop_words=stop_words,
        ngram_range=ngram_range
    )
    vector_count = vectorizer.fit_transform(content)
    terms = vectorizer.get_feature_names()
    freqs = vector_count.sum(axis=0).A1
    n_gram_count = dict(zip(terms, freqs))
    #n_gram_count = [(w, n_gram_count[w]) for w in sorted(n_gram_count, key=n_gram_count.get, reverse=True)]
    n_gram_count_data = {}
    [n_gram_count_data.setdefault(len(w.split()),[]).append((w,n_gram_count[w])) for w in sorted(n_gram_count, key=n_gram_count.get, reverse=True)]
    return n_gram_count_data


def n_gram_analyzer(content, ngram_range=(1,5), stop_words=stop):
    vectorizer = CountVectorizer(
        tokenizer=pre_process,
        lowercase=False,
        stop_words=stop_words,
        ngram_range=ngram_range
    )
    vector_count = vectorizer.fit_transform(content)
    terms = vectorizer.get_feature_names()
    freqs = vector_count.sum(axis=0).A1
    n_gram_count = dict(zip(terms, freqs))
    #n_gram_count = [(w, n_gram_count[w]) for w in sorted(n_gram_count, key=n_gram_count.get, reverse=True)]
    n_gram_count_data = {}
    [n_gram_count_data.setdefault(len(w.split()),[]).append((w,n_gram_count[w])) for w in sorted(n_gram_count, key=n_gram_count.get, reverse=True)]
    tf_transformer = TfidfTransformer(use_idf=False).fit(vector_count)
    tf_count = tf_transformer.transform(vector_count)
    tf_freqs = tf_count.sum(axis=0).A1
    n_gram_tf = dict(zip(terms, tf_freqs))
    #n_gram_tf = [(w, n_gram_tf[w]) for w in sorted(n_gram_tf, key=n_gram_tf.get, reverse=True)]
    n_gram_tf_data = {}
    [n_gram_tf_data.setdefault(len(w.split()),[]).append((w,n_gram_tf[w])) for w in sorted(n_gram_tf, key=n_gram_tf.get, reverse=True)]
    tf_idf_transformer = TfidfTransformer()
    tf_idf_count = tf_idf_transformer.fit_transform(vector_count)
    tf_idf_freqs = tf_idf_count.sum(axis=0).A1
    n_gram_tf_idf = dict(zip(terms, tf_idf_freqs))
    n_gram_tf_idf_data = {}
    [n_gram_tf_idf_data.setdefault(len(w.split()),[]).append((w,n_gram_tf_idf[w])) for w in sorted(n_gram_tf_idf, key=n_gram_tf_idf.get, reverse=True)]
    #n_gram_tf_idf = [(w, n_gram_tf_idf[w]) for w in sorted(n_gram_tf_idf, key=n_gram_tf_idf.get, reverse=True)]
    #vocabulary = count_vect.get_feature_names()
    #Computa la matriz para el conteo
    #vocabulary_count = np.sum(tf_idf_count.toarray(), axis=0)
    #for tag, count in zip(terms, vocabulary_count):
    #    print count, tag
    #Presenta la relación de palabras y frecuencias
    #metodo 1
    #[(terms[n], tf_idf_count.toarray()[0][n])for n in range(0,len(terms))]
    #metodo 2
    #zip(terms.get_feature_names(),np.asarray(tf_idf_count.sum(axis=0)).ravel())
    #muestra el peso de los términos
    #tf_idf_transformer.idf_
    #return n_gram_count, n_gram_tf, n_gram_tf_idf
    return n_gram_count_data, n_gram_tf_data, n_gram_tf_idf_data


def tagging(content):
    result = tagger.tag_sents(content)
    #print content
    #print dir(tagger)
    return result


def word2vec(words, method='similarity', output=10):
    if method == 'similarity':
        result = model.most_similar(words, topn=output)
    else:
        result = model.doesnt_match(words)
    #print content
    #print dir(tagger)
    return result


def make_graphic(word_freq, path='static/term_freq.json'):
    labels, freq = zip(*word_freq)
    data = {'vals': list(freq), 'index': list(labels)}
    bar = vincent.Bar(data, iter_idx='index')
    bar.axis_titles(y=u'Veces', x=u'Términos')
    bar.to_json(path)


def sentiment_analysis(text):
    parts = text.split(' ')
    if len(parts) < 3:
        prediction = ['neu']
    else:
        data = sentiment_vectorizer.transform([text])
        prediction = sentiment_classifier.predict(data)
    return prediction[0]


def topic_analysis(text):
    data = [text]
    prediction = topic_classifier.predict(data)
    return topic_headers[prediction]