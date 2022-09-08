import feedparser
import requests

print("Fetching news...")

feed_urls = []
with open('feeds.txt', 'r') as f:
    feed_urls = f.readlines()

feeds = []
for feed_url in feed_urls:
    feed = feedparse.parse_feed(feed_url)
    print(feed)
    feeds.append(feed)
