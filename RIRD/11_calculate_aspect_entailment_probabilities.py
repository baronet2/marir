multi_aspect_queries = {
    'Italian place with a burger': ['Italian place', 'burger'],
    'A cafe that also offers beer': ['cafe', 'beer'],
    'Japanese restaurant with pasta': ['Japanese restaurant', 'pasta'],
    'An ice cream shop with bubble tea': ['ice cream shop', 'bubble tea'],
    'I am in search of a fancy Pakistani restaurant with authentic food': ['fancy', 'Pakistani restaurant']
}
aspects = [x for xs in multi_aspect_queries.values() for x in xs]

import pandas as pd
rird_reviews = (
    pd.read_csv(
        'https://raw.githubusercontent.com/D3Mlab/rir/main/data/50_restaurants_all_rates.csv',
        usecols = ["name", "review_text"]
    )
    .groupby('name')
    .head(5)
)

# See https://huggingface.co/facebook/bart-large-mnli
from transformers import pipeline
classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")

print(f"Calculating entailment probabilities for {len(rird_reviews)} reviews and {len(aspects)} aspects...")
entailment_probabilities = classifier(list(rird_reviews['review_text']), aspects, multi_label = True)

import json
with open("aspect_entailment_probabilities.json", "w") as f:
  json.dump(entailment_probabilities, f)