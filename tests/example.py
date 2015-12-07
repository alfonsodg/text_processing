from text_processing import *

if __name__ == '__main__':
    import pymongo
    columns = [
        '_id', 'id', 'text', 'favorite_count', 'retweet_count', 'lang'
    ]

    query_cols = list_to_dict(columns)
    client = pymongo.MongoClient("localhost", 27017)
    db_aux = client.trending
    media_sources_twitter = db_aux.media_sources_twitter_contents
    media_sources_twitter_demographics = db_aux.media_sources_twitter_demographics
    data = media_sources_twitter.find({'retweet_count': {'$gt': 10}}, query_cols)
    tweets = []
    for tweet in data[:100000]:
        if tweet['text'].find('RT') >= 0 or tweet['text'].find('recetuit') >= 0:
            continue
        txt = tweet['text']
        id = tweet['id']
        tweets.append(txt)
    # tweets = [
    #     u"RT @marcobonzanini: solo un ejemplo! :D http://example.com #NLP",
    #     u"Solo una vez más es otro ejemplo de que jamás hay que creerte, gracias :) #Python",
    #     u"Definitivamente es un gracias :) #Python todo el día"
    #     ]
    #print tokenize_stem(tweets[0])
    #terms = uni_gram_finder(tweets)
    #uni, bi, tri = n_gram_nltk(terms)
    #make_graphic(uni)
    ###Obtiene palabras tokenizadas y oraciones con tokens
    n_terms, s_terms = terms_builder(tweets)
    tagging(s_terms)
    ###Cuenta n-gramas
    n_gram_count = n_gram_counter(tweets)
    ###Obtiene palabras tokenizadas con raíz gramatical, interpolado a oraciones
    sn, ss = terms_builder(tweets, stop_words=None, stem=True)
    #print s_terms
    # for pal in count[:1000]:
    #     cnt = pal.items()
    #     #print cnt
    #     tex = cnt[0][0].split()
    #     cnx = cnt[0][1]
    #     if cnx > 3 and len(tex) >= 3:
    #         print cnt[0][0], cnx
    # s_terms = s_terms[:10]
    # print tagging(s_terms)
    # #print tokenize_stem(tweets[0])
    # print s_terms
    # ss = ss[:10]
    # print
    # print ss
    #    chunking(tweets[0])
    #    res = topic(u'Uno de los sistemas operativos para móbiles más extendido es android')
    #    print(res)


# import gensim, logging
# logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
#
# sentences = [['first', 'sentence'], ['second', 'sentence']]
# model = gensim.models.Word2Vec(sentences, min_count=1)
tweets = [
    u"RT @marcobonzanini: solo un ejemplo! :D http://example.com #NLP",
    u"Solo una vez más es otro ejemplo de que jamás hay que creerte, gracias :) #Python",
    u"Definitivamente es un gracias :) #Python todo el día"
    ]