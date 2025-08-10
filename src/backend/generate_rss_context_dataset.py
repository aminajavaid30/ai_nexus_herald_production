import feedparser
import json
from typing import List
import os
from src.backend.utils import load_yaml_config
from src.backend.paths import RSS_CONFIG_FPATH, DATA_DIR

rss_config = load_yaml_config(RSS_CONFIG_FPATH)
rss_feeds = rss_config["rss_feeds"]

# Extract RSS feed URLs
RSS_FEEDS = [feed["url"] for feed in rss_feeds.values()]

def fetch_rss_titles_and_articles(rss_urls: List[str], max_items: int = 30):
    context_data = []

    for url in rss_urls:
        feed = feedparser.parse(url)
        for entry in feed.entries:
            context_data.append({
                "title": entry.title,
                "link": entry.link,
                "summary": entry.get("summary", ""),
                "content": entry.get("content", ""),
            })

    return context_data

def save_dataset(context_data, output_path="rss_context_dataset.json"):
    # Create the output directory if it doesn't exist
    output_dir = os.path.dirname(output_path)
    os.makedirs(output_dir, exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump({"rss_context": context_data}, f, indent=2)

if __name__ == "__main__":
    data = fetch_rss_titles_and_articles(RSS_FEEDS)
    path = DATA_DIR + "/rss_context_dataset.json"
    save_dataset(data, path)