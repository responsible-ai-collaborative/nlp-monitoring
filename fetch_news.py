MOCK = True
aws_root = 'https://q3z6vr2qvj.execute-api.us-west-2.amazonaws.com'

with open('data_keywords.txt', 'r') as f:
  keywords  = [l for l in f.read().split('\n') if len(l.strip()) > 0]

with open('data_feeds.txt', 'r') as f: 
  feed_urls = [l for l in f.read().split('\n') if len(l.strip()) > 0]

def main(
    keywords = keywords, 
    feed_urls = feed_urls, 
    connection_string = None,
    upload = True,
    force = False,
    mock = MOCK,
    seconds_between_requests = 2
):
    print("Fetching news...")
  
    mongo_client = get_mongo_client(connection_string)

    mean_embedding = get_mean_embedding(mongo_client)

    stemmer = PorterStemmer()
    stemmed_keywords = get_stemmed_keywords(keywords, stemmer=stemmer)

    entities = get_entities(mongo_client)
    stemmed_entities = get_stemmed_keywords(entities, stemmer=stemmer)
        
    for feed_url in feed_urls:
        feed = feedparser.parse(feed_url)
        last_hit = 0
        for entry in feed['entries']:
            now = time.time()
            delta = now - last_hit
            if delta < seconds_between_requests:
                time.sleep(seconds_between_requests - delta)
            if process_url(
                entry['link'], 
                mean_embedding=mean_embedding, 
                mongo_client=mongo_client, 
                keywords = keywords,
                stemmed_keywords=stemmed_keywords,
                stemmer=stemmer,
                upload=upload,
                mock=mock,
                force=force,
                entities=entities,
                stemmed_entities=stemmed_entities,
            ):
                last_hit = now

def get_entities(mongo_client = None, connection_string = None):
    if not mongo_client:
        mongo_client = get_mongo_client(connection_string)
    if not mongo_client:
        raise Exception("No connection string provided")
    incidents_collection = mongo_client['aiidprod'].incidents

    entity_fields = [
        'Alleged deployer of AI system',
        'Alleged developer of AI system',
        'Alleged harmed or nearly harmed parties',
    ]

    entities = set()

    projection = {}
    for field in entity_fields:
        projection[field] = True

    for incident in incidents_collection.find({}, projection):
        for field in entity_fields:
            for entity in incident[field]:
                entities.add(entity)

    return entities 

def get_mongo_client(connection_string = None):
    if not connection_string:
        connection_string = environ.get('MONGODB_CONNECTION_STRING')
    if connection_string:
        return MongoClient(connection_string)
 
def get_mean_embedding(mongo_client = None, connection_string = None):
    if not mongo_client:
        mongo_client = get_mongo_client(connection_string)
    if not mongo_client:
        raise Exception("No connection string provided")

    incidents_collection = mongo_client['aiidprod'].incidents
    m = np.array([0] * 768)
    for offset, incident in enumerate(incidents_collection.find(
        { 'embedding': { '$exists': True } }, 
        { 'embedding': { 'vector': True } }
    )):
        i = offset + 1

        # As the number of vectors contributing to the mean increases,
        # the contribution of each one decreases:
        #
        # i=1 → m = (0/1)m + (1/1)v
        # i=2 → m = (1/2)m + (1/2)v
        # i=3 → m = (1/3)m + (2/3)v
        # i=4 → m = (3/4)m + (1/4)v
        # ...
        v = np.array(incident['embedding']['vector'])
        m = ( v * 1 / i) + (m * (i - 1) / i)

    return m


def process_url(
    article_url,
    mean_embedding = [0] * 768,
    keywords = keywords,
    connection_string = None,
    mongo_client = None, 
    stemmed_keywords = None,
    stemmer = None,
    upload = False,
    force = False,
    mock = True,
    entities=None,
    stemmed_entities=None,
):
    if not mongo_client: mongo_client = get_mongo_client(connection_string)
    candidates_collection = None
    if mongo_client:
        candidates_collection = mongo_client['aiidprod'].candidates
        if all([e == 0 for e in mean_embedding]):
            mean_embedding = get_mean_embedding(mongo_client=mongo_client)


    if not stemmer: stemmer = PorterStemmer()
    if not stemmed_keywords: 
        stemmed_keywords = get_stemmed_keywords(keywords, stemmer=stemmer)
    if not stemmed_entities: 
        stemmed_entities= get_stemmed_keywords(entities, stemmer=stemmer)

    try:
        print("\nFetching", article_url)

        if mongo_client != None:
            result = candidates_collection.find_one({ 'url': article_url })
            if result is not None and not force:
                print('URL already processed. Skipping...')
                return False

        article = get_article(article_url)
        if not article: return True
        print(article['title'])
        print(article['date_published'])

        article_words = [
            stemmer.stem(word).lower() for word in
            word_tokenize(article['plain_text'])
        ]

        matching_keywords = [ 
            word.strip() for word in keywords 
            if stemmed_keywords[word].lower() in ' '.join(article_words)
        ]
        article['matching_keywords'] = matching_keywords
        print('Matching Keywords:', article['matching_keywords'])
        
        matching_entities = [ 
            word.strip() for word in entities 
            if ' ' + (
                word.lower() if word in ['ETS'] else 
                stemmed_entities[word].lower()
            ) + ' ' in ' '.join(article_words)
        ]
        article['matching_entities'] = matching_entities
        print('Matching Entities:', article['matching_entities'])

        if (len(article['matching_keywords']) > 0) and not all([e == 0 for e in mean_embedding]):
            print('Running NLP...')
            
            nlp_response = None
            if mock:
                with open('data_mock.json') as f: 
                    nlp_response = json.load(f) 
            else:
                # nlp_response = requests.post(
                #     aws_root + '/text-to-embed',
                #     data = {
                #         'text': article['plain_text']
                #     }
                # )
                nlp_response = requests.get(
                    (
                        aws_root + '/text-to-embed?' +
                        urllib.parse.urlencode({
                            'text': article['plain_text'],
                        })
                    )[0:2048],
                    timeout=10
                ).json()

            article['embedding'] = nlp_response['body']['embedding']

            article['similarity'] = cosine_similarity(
                article['embedding']['vector'],
                mean_embedding
            )
            print('Mock Similarity' if mock else 'Similarity:', article['similarity'])

            article['match'] = True

            if mongo_client and upload:
                print('Uploading to MongoDB...')
                candidates_collection.insert_one(article)
        else:
            if mongo_client and upload:
                print('Uploading to MongoDB...')
                candidates_collection.insert_one({ 
                    'match': False, 
                    'url': article['url'], 

                    # Keeping the title will help us see
                    # if things that should be accepted are rejected.
                    'title': article['title']
                })
        return True

    except Exception as ex:
        traceback.print_exception(type(ex), ex, ex.__traceback__)
        return True
    


def get_article(article_url):

    plaintext_reader = HTML2Text()
    plaintext_reader.ignore_links = True
    plaintext_reader.ignore_images = True

    article_response = requests.get(article_url, timeout=10)

    article_doc = Document(article_response.text)

    summary = article_doc.summary()

    date_published = htmldate.find_date(article_response.text)

    article = {
        'title': article_doc.title(),
        'text': html2text(summary),
        'plain_text': plaintext_reader.handle(summary),
        'url': article_url,
        'date_published': date_published
    }

    for key in article.keys():
        if not article[key]:
          return None

    return article

def get_stemmed_keywords(keywords, stemmer=None):
    if not stemmer:
        stemmer = PorterStemmer()
    stemmed_keywords = {}
    for keyword in keywords:
        stemmed_keywords[keyword] = ' '.join([
            stemmer.stem(token) for token in keyword.split(' ')
        ])
    return stemmed_keywords

cosine_similarity = lambda a, b: np.dot(a, b) / (norm(a) * norm(b))


import feedparser
import requests
import urllib
import json
import numpy as np
import htmldate
import traceback
import time
from pymongo import MongoClient
from html2text import html2text, HTML2Text
from readability import Document
from os import environ
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
from numpy.linalg import norm

if __name__ == "__main__": main()
