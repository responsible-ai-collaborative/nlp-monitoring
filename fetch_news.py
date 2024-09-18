aws_root = 'https://q3z6vr2qvj.execute-api.us-west-2.amazonaws.com'
ignored_entities = ['Cruise', 'Grab' ]

with open('data_keywords.txt', 'r') as f:
    keywords  = [l for l in f.read().split('\n') if len(l.strip()) > 0]

with open('data_harm_keywords.txt', 'r') as f:
    harm_keywords  = [l for l in f.read().split('\n') if len(l.strip()) > 0]

def main(
    keywords = keywords, 
    connection_string = None,
    upload = True,
    force = False,
    mock = False,
    seconds_between_requests = 2
):
    print("Fetching news...")

    with open('data_feeds.json', 'r') as f: 
        feeds_config = json.load(f)
  
    mongo_client = get_mongo_client(connection_string)

    mean_embedding = get_mean_embedding(mongo_client=mongo_client)

    stemmer = PorterStemmer()
    stemmed_keywords = get_stemmed_keywords(keywords, stemmer=stemmer)

    entities = get_entities(mongo_client)
    stemmed_entities = get_stemmed_keywords(entities, stemmer=stemmer)

    stemmed_harm_keywords = get_stemmed_keywords(harm_keywords, stemmer=stemmer)
        
    for config in feeds_config:
        feed_url = config['url']
        feed = feedparser.parse(feed_url)
        last_hit = 0
        for entry in feed['entries']:
            text = entry['summary'] if config.get('fulltext') else None
            now = time.time()
            delta = now - last_hit
            if delta < seconds_between_requests:
                time.sleep(seconds_between_requests - delta)
            if process_url(
                entry['link'], 
                text=text,
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
                stemmed_harm_keywords=stemmed_harm_keywords,
            ):
                last_hit = now

    if mongo_client:
        delete_old_articles(mongo_client)
        trim_old_articles(mongo_client)

def get_entities(mongo_client = None, connection_string = None):
    if not mongo_client:
        mongo_client = get_mongo_client(connection_string)

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
                if not entity in ignored_entities:
                    entities.add(entity)

    return entities 

def get_mongo_client(connection_string = None, required = False):
    connection_string = connection_string or environ.get('MONGODB_CONNECTION_STRING')
    if not connection_string:
        if required:
            raise Exception("No connection string provided")
        else:
            return None
    return MongoClient(connection_string)
 
def get_mean_embedding(mongo_client = None, connection_string = None):
    if not mongo_client:
        mongo_client = get_mongo_client(connection_string)
    
    query = { 'embedding': { '$exists': True } }

    incidents_collection = mongo_client['aiidprod'].incidents
    m = np.array([0] * 768)
    for offset, incident in enumerate(incidents_collection.find(
        query, 
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

def delete_old_articles(mongo_client):
    candidates_collection = mongo_client['aiidprod'].candidates

    removal_cutoff_date = (
      datetime.datetime.now() - datetime.timedelta(days=90)
    ).isoformat()[0:10]

    candidates_collection.delete_many({
        '$or': [
            {'date_scraped':   {'$lt': removal_cutoff_date }},
            {'date_published': {'$lt': removal_cutoff_date }},
            {
                '$and': [
                    {'date_scraped':   {'$exists': False }},
                    {'date_published': {'$exists': False }},
                ]
            }
        ]
    })


def trim_old_articles(mongo_client):
    candidates_collection = mongo_client['aiidprod'].candidates

    for article in candidates_collection.find({
        '$or': [
            {'text':       {'$exists': True}},
            {'embedding':  {'$exists': True}},
            {'plain_text': {'$exists': True}},
        ]
    }):
        try:
            article_date = None
            if article.get('date_published'):
                try:
                    article_date = dateutil.parser.parse(article['date_published'])
                except:
                    pass
                if not article_date and article.get('date_scraped'):
                    try:
                        article_date = dateutil.parser.parse(article['date_scraped'])
                    except:
                        pass
                if not article_date:
                    # Date at which date_scraped started being collected.
                    article_date = dateutil.parser.parse('2023-08-30') 
                  
                article_age = datetime.datetime.now() - article_date
                if article.get('text') and article_age.days > 30:
                    candidates_collection.update_one(
                        { 'url': article['url'] },
                        {'$unset': {'text': '', 'plain_text': '', 'embedding': ''}}
                    )
        except Exception as ex:
            traceback.print_exception(type(ex), ex, ex.__traceback__)

def process_url(
    article_url,
    mean_embedding=[0] * 768,
    keywords=keywords,
    harm_keywords=harm_keywords,
    text=None,
    connection_string = None,
    mongo_client = None, 
    stemmer = None,
    upload = False,
    force = False,
    mock = True,
    entities=[],
    stemmed_keywords = None,
    stemmed_entities=None,
    stemmed_harm_keywords=None,
):
    if not mongo_client: mongo_client = get_mongo_client(connection_string, required = False)
    candidates_collection = None
    if mongo_client:
        candidates_collection = mongo_client['aiidprod'].candidates
        if all([e == 0 for e in mean_embedding]):
            mean_embedding = get_mean_embedding(mongo_client=mongo_client)


    if not stemmer: stemmer = PorterStemmer()
    if not stemmed_keywords: 
        stemmed_keywords = get_stemmed_keywords(keywords, stemmer=stemmer)
    if not stemmed_harm_keywords: 
        stemmed_harm_keywords = get_stemmed_keywords(harm_keywords, stemmer=stemmer)
    if not stemmed_entities: 
        stemmed_entities= get_stemmed_keywords(entities, stemmer=stemmer)

    try:
        print("\nFetching", article_url)

        if mongo_client != None:
            article = candidates_collection.find_one({ 'url': article_url })
            if article is not None and not force:
                print('URL already processed. Skipping...')
                return False

        article = get_article(article_url, text=text)
        if not article: 
            print("Could not get article")
            return False
        print("Title:", article['title'])
        print("Date Published:", article['date_published'])

        article_words = [
            stemmer.stem(word).lower() for word in
            word_tokenize(article['plain_text'])
        ]

        matching_keywords = [ 
            word.strip() for word in keywords 
            if stemmed_keywords[word].lower() in ' '.join(article_words)
        ]
        article['matching_keywords'] = matching_keywords
        print('AI Keywords:', article['matching_keywords'])
        
        matching_harm_keywords = [ 
            word.strip() for word in harm_keywords
            if ' ' + (stemmed_harm_keywords.get(word) or "").lower() + ' ' in ' '.join(article_words)
        ]
        article['matching_harm_keywords'] = matching_harm_keywords
        print('Harm Keywords:', article['matching_harm_keywords'])

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

                    'date_published': article['date_published'],
                    'date_scraped': article['date_scraped'],
                })
        return article

    except Exception as ex:
        traceback.print_exception(type(ex), ex, ex.__traceback__)
        return False

def get_article(article_url, text=None):
    try:
        mercury_output = json.loads(subprocess.check_output([
          './node_modules/@postlight/mercury-parser/cli.js',
          '--format', 'markdown',
          article_url
        ]).decode('utf-8'))

        markdown = (
          mercury_output['content']
            .replace('\n\nAdvertisement\n\n', '\n\n') # arstechnica
        )

        plain_text = subprocess.check_output(
          ['pandoc', '--from', 'markdown', '--to', 'plain'], 
          input=markdown, 
          encoding='utf-8'
        )

        article = {
            'title': mercury_output.get('title'),
            'text': markdown,
            'plain_text': plain_text,
            'url': article_url,
            'date_published': (mercury_output.get('date_published') or "")[0:10],
            'date_scraped': datetime.datetime.now().isoformat()[0:10],
        }

        if not article.get('text'):
            return None

        return article

    except Exception as ex:
        traceback.print_exception(type(ex), ex, ex.__traceback__)
        return None 

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
import numpy as np
import traceback
import time
import json
import subprocess
import datetime
import dateutil.parser
from pymongo import MongoClient
from os import environ
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
from numpy.linalg import norm

if __name__ == "__main__": main()
