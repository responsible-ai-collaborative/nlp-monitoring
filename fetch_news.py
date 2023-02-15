MOCK = True
aws_root = 'https://q3z6vr2qvj.execute-api.us-west-2.amazonaws.com'

with open('data_keywords.txt', 'r') as f:
    keywords  = [l for l in f.read().split('\n') if len(l.strip()) > 0]

with open('data_harm_keywords.txt', 'r') as f:
    harm_keywords  = [l for l in f.read().split('\n') if len(l.strip()) > 0]

ignored_entities = [
    'Cruise',
    'Grab',
]

def main(
    keywords = keywords, 
    connection_string = None,
    upload = True,
    force = False,
    mock = MOCK,
    seconds_between_requests = 2
):
    print("Fetching news...")

    with open('data_feeds.json', 'r') as f: 
        feeds_config = json.load(f)
  
    mongo_client = get_mongo_client(connection_string)

    mean_embedding = get_mean_embedding(mongo_client=mongo_client)

    classifications = [
        ['CSET', 'Sector of Deployment', 'Transportation and storage'],
        ['CSET', 'Harm Type', 'Harm to social or political systems'],
    ]
    classification_mean_embeddings = {}
    for classification in classifications:
        classification_mean_embeddings[
            ':'.join(classification)
        ] = get_mean_embedding(mongo_client=mongo_client, classification=classification)

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
                classification_mean_embeddings=classification_mean_embeddings,
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
                if not entity in ignored_entities:
                    entities.add(entity)

    return entities 

def get_mongo_client(connection_string = None):
    if not connection_string:
        connection_string = environ.get('MONGODB_CONNECTION_STRING')
    if connection_string:
        return MongoClient(connection_string)
 
def get_mean_embedding(mongo_client = None, connection_string = None, classification = None):
    if not mongo_client:
        mongo_client = get_mongo_client(connection_string)
    if not mongo_client:
        raise Exception("No connection string provided")
    
    query = { 'embedding': { '$exists': True } }

    if classification:
        classifications_collection = mongo_client['aiidprod'].classifications
        key = 'classifications.' + classification[1]
        incident_ids = [
            incident['incident_id'] for incident in classifications_collection.find(
                { 
                    'namespace': classification[0], 
                    '$or': [
                        {key: {'$elemMatch': { '$eq' : classification[2] }}},
                        {key: classification[2]},
                    ]
                }, 
                {'incident_id': True}
            )
        ]
        query['incident_id'] = {'$in': incident_ids}

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


def process_url(
    article_url,
    mean_embedding = [0] * 768,
    classification_mean_embeddings=None,
    keywords = keywords,
    text=None,
    connection_string = None,
    mongo_client = None, 
    stemmed_keywords = None,
    stemmer = None,
    upload = False,
    force = False,
    mock = True,
    entities=None,
    stemmed_entities=None,
    stemmed_harm_keywords=None,
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
    if not stemmed_keywords: 
        stemmed_harm_keywords = get_stemmed_keywords(harm_keywords, stemmer=stemmer)
    if not stemmed_entities: 
        stemmed_entities= get_stemmed_keywords(entities, stemmer=stemmer)

    try:
        print("\nFetching", article_url)

        if mongo_client != None:
            result = candidates_collection.find_one({ 'url': article_url })
            if result is not None and not force:
                print('URL already processed. Skipping...')
                return False

        article = get_article(article_url, text=text)
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
        print('AI Keywords:', article['matching_keywords'])
        
        matching_harm_keywords = [ 
            word.strip() for word in harm_keywords
            if ' ' + stemmed_harm_keywords[word].lower() + ' ' in ' '.join(article_words)
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

            if classification_mean_embeddings:
                article['classification_similarity'] = []
                for key in classification_mean_embeddings.keys():
                    article['classification_similarity'].append({
                        'classification': key, 
                        'similarity': cosine_similarity(
                            article['embedding']['vector'],
                            classification_mean_embeddings[key]
                        )
                    })

            print('Mock Similarity' if mock else 'Similarity:', article['similarity'])
            print('Mock Classifiction Similarity' if mock else 'Classification Similarity:', article['classification_similarity'])

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
            'url': mercury_output.get('url') or article_url,
            'date_published': mercury_output.get('date_published')
        }

        for key in article.keys():
            if not article[key]:
              return None

        return article

    except:
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
from pymongo import MongoClient
from os import environ
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
from numpy.linalg import norm

if __name__ == "__main__": main()
