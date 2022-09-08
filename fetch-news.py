import feedparser
import requests
import urllib
import json
import numpy as np
from pymongo import MongoClient
from html2text import html2text, HTML2Text
from readability import Document
from os import environ
from scipy import spatial
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer

MONGODB_URI = environ['MONGODB_CONNECTION_STRING']
MOCK = True 

plaintext_reader = HTML2Text()
plaintext_reader.ignore_links = True
plaintext_reader.ignore_images = True

print("Fetching news...")

with open('feeds.txt', 'r') as f:
    feed_urls = f.readlines()

mongo_client = MongoClient(MONGODB_URI)
incidents_collection = mongo_client['aiidprod'].incidents

stemmer = PorterStemmer()

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

mean_embedding = m

print(mean_embedding)

for feed_url in feed_urls:
    feed = feedparser.parse(feed_url)
    for entry in feed['entries']:
        try:
            article_url = entry['link']

            print("\nFetching", article_url)
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

            matching_keywords = [ word for word in [
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
            
            print(matching_keywords)


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
                        aws_root + '/text-to-embed?' +
                        urllib.parse.urlencode({
                            'text': article['plain_text'],
                        }),
                        timeout=10
                    ).json()

                article['embedding'] = nlp_response['body']['embedding']

                article['similarity'] = spatial.distance.cosine(
                    article['embedding']['vector'],
                    mean_embedding
                )

                print(article['similarity'])
        except Exception as e:
            print(e)
            continue


