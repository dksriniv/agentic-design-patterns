"""Offline, deliberately simple FAQ retrieval. No embeddings required."""
import json
import re
from pathlib import Path

STOP_WORDS = {"a", "an", "the", "i", "my", "is", "to", "how", "do", "can", "of", "and", "for", "in", "it", "with"}


def tokens(text):
    return set(re.findall(r"[a-z0-9]+", text.lower())) - STOP_WORDS


class FAQSearch:
    def __init__(self):
        self.documents = json.loads(Path(__file__).with_name("knowledge_base.json").read_text())

    def search(self, query):
        words = tokens(query)
        scored = [(len(words & tokens(doc["title"] + " " + doc["text"])), doc) for doc in self.documents]
        ranked = sorted(scored, key=lambda pair: (-pair[0], pair[1]["id"]))
        return {"documents": [doc for score, doc in ranked if score > 0][:3]}
