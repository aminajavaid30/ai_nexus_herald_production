from langchain_core.tools import tool
import feedparser
import datetime
import os
from src.backend.paths import OUTPUTS_DIR

from sentence_transformers import SentenceTransformer, util

model = SentenceTransformer("nomic-ai/nomic-embed-text-v1", trust_remote_code=True) # lightweight and fast

@tool("extract_titles_from_rss", return_direct=True)
def extract_titles_from_rss(urls: list[str]) -> list[str]:
    """Extracts titles from RSS feeds."""
    titles = []
    for url in urls:
        feed = feedparser.parse(url)
        for entry in feed.entries:
            if 'title' in entry:
                titles.append(entry.title)
    return titles

@tool("extract_news_from_rss", return_direct=True)
def extract_news_from_rss(feed_urls: list[str], topic: str, threshold: float = 0.7):
    """Extracts news articles from RSS feeds relevant to a single topic using embeddings."""
    topic_articles = []

    topic_embedding = model.encode(topic, convert_to_tensor=True)

    for url in feed_urls:
        feed = feedparser.parse(url)
        for entry in feed.entries:
            title = entry.get('title', '')
            link = entry.get('link', '')
            summary = entry.get('summary', '') or entry.get('description', '')

            raw_content = entry.get('content')
            if isinstance(raw_content, list) and raw_content:
                content = raw_content[0].get('value', '')
            elif isinstance(raw_content, str):
                content = raw_content
            else:
                content = ''

            article_text = title + " " + summary
            article_embedding = model.encode(article_text, convert_to_tensor=True)

            score = util.cos_sim(article_embedding, topic_embedding).item()

            # Replace double quotes inside title, summary, and content with single quotes
            title = title.replace('"', "'")
            summary = summary.replace('"', "'")
            content = content.replace('"', "'")
            
            if score >= threshold:
                topic_articles.append({
                    "title": title,
                    "link": link,
                    "summary": summary,
                    "content": content,
                    "similarity": score
                })

    # Sort articles by similarity score
    topic_articles.sort(key=lambda x: x["similarity"], reverse=True)

    # Select top 1 article based on similarity score - due to LLM rate limits
    if len(topic_articles) > 1:
        topic_articles = topic_articles[:1]

    return topic_articles

@tool("save_newsletter", return_direct=True)
def save_newsletter(newsletter: str):
    """Saves a newsletter to a file."""

    # get current date and time
    now = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    news_dir = os.path.join(OUTPUTS_DIR, "newsletters")
    os.makedirs(news_dir, exist_ok=True)

    filename = f"{news_dir}/newsletter_{now}.md"
    with open(filename, "w", encoding="utf-8") as f:
        f.write(newsletter)