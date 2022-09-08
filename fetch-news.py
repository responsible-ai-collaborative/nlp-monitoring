import feedparser
import requests

print("Fetching news...")

feeds = []
with open('feeds.txt', 'r') as f:
    feeds = f.readlines()
