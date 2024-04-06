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

from sentence_transformers import SentenceTransformer, util
model = SentenceTransformer('sentence-transformers/msmarco-distilbert-base-tas-b')

query_embs = model.encode(aspects)
review_embs = model.encode(list(rird_reviews['review_text']))

out = util.dot_score(query_embs, review_embs)
scores = (
    pd.DataFrame(out, columns = rird_reviews['review_text'], index = aspects)
    .reset_index()
    .pipe(pd.melt, id_vars = "index")
    .rename(columns = {"index": "aspect", "value": "score"})
)

scores.to_csv("aspect_dense_scores.csv")