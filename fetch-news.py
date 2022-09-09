import feedparser
import requests
import urllib
import json
import numpy as np
from pymongo import MongoClient
from html2text import html2text, HTML2Text
from readability import Document
from os import environ
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
from numpy.linalg import norm

cosine_similarity = lambda a, b: np.dot(a, b)/(norm(a)*norm(b))

MONGODB_URI = environ.get('MONGODB_CONNECTION_STRING')
MOCK = False

plaintext_reader = HTML2Text()
plaintext_reader.ignore_links = True
plaintext_reader.ignore_images = True

print("Fetching news...")

with open('feeds.txt', 'r') as f:
    feed_urls = f.readlines()

stemmer = PorterStemmer()

m = np.array([0] * 768)

mongo_client = None
incidents_collection = None
candidates_collection = None
if MONGODB_URI:
    mongo_client = MongoClient(MONGODB_URI)
    incidents_collection = mongo_client['aiidprod'].incidents
    candidates_collection = mongo_client['aiidprod'].candidates

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

mean_embedding = m
    
for feed_url in feed_urls:
    feed = feedparser.parse(feed_url)
    for entry in feed['entries']:
        try:
            article_url = entry['link']
            
            print("\nFetching", article_url)

            if mongo_client:
                result = candidates_collection.find_one({ 'url': article_url })
                if result is not None:
                    print('URL already processed. Skipping...')
                    continue

            article_response = requests.get(article_url, timeout=10)

            article_doc = Document(article_response.text)

            summary = article_doc.summary()

            article = {
                'title': article_doc.title(),
                'text': html2text(summary),
                'plain_text': plaintext_reader.handle(summary),
                'url': article_url
            }
            print(article['title'])

            article_words = [
                stemmer.stem(word).lower() for word in
                word_tokenize(article['plain_text'])
            ]

            matching_keywords = [ word.strip() for word in [
                ' AI ',
                'artificial intelligence',
                'machine learn',
                'deep learning',
                'facial recognition',
                'face detection',
                'object detection',
                'object recognition',
                'computer vision',
                'self driving',
                'self-driving',
                'autonomous vehicle',
                'autonomous driving',
                'algorithmic bias',
                'algorithmic fairness',
                'biased algorithm',
                'language model',
                'neural net',
                'language processing',
                'NLP',
            ] if stemmer.stem(word).lower() in ' '.join(article_words)]
            
            print('Matching Keywords:', matching_keywords)

            article['matching_keywords'] = matching_keywords


            if (len(matching_keywords) > 0):
                print('Running NLP...')
                aws_root = 'https://q3z6vr2qvj.execute-api.us-west-2.amazonaws.com';
                
                nlp_response = None

                if MOCK:
                    with open('mock.json') as f: 
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

                if not all(mean_embedding == 0):
                    article['similarity'] = cosine_similarity(
                        article['embedding']['vector'],
                        mean_embedding
                    )
                    print(article['similarity'])

                    article['match'] = True

                    if mongo_client:
                        print('Uploading to MongoDB...')
                        candidates_collection.insert_one(article)
            else:
                if mongo_client:
                    print('Uploading to MongoDB...')
                    candidates_collection.insert_one({ 
                        'match': False, 
                        'url': article['url'], 

                        # Keeping the title will help us see
                        # if things that should be accepted are rejected.
                        'title': article['title']
                    })
                

        except Exception as e:
            print(e)
            continue


