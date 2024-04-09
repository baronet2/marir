import pandas as pd
recipe_mpr = pd.read_json('https://raw.githubusercontent.com/D3Mlab/Recipe-MPR/main/data/500QA.json', orient = 'records')

recipe_mpr_clean = (
    recipe_mpr
    .assign(
        query_aspects = lambda d: d.correctness_explanation.apply(lambda x: list(x.keys())),
        num_aspects = lambda d: d.query_aspects.apply(len),
        option_id = lambda d: d.options.apply(lambda x: list(x.keys())),
        option_text = lambda d: d.options.apply(lambda x: list(x.values()))
    )
    .query("num_aspects > 1")
    .explode(['option_id', 'option_text'])
    [['query', 'option_id', 'option_text']]
)

from sentence_transformers import SentenceTransformer, util
model = SentenceTransformer('sentence-transformers/msmarco-distilbert-base-tas-b')

queries = []
items = []
scores = []
for name, group in recipe_mpr_clean.groupby('query'):
    queries.append(name)
    items.append(list(group['option_text']))

    query_emb = model.encode(name)
    item_embs = model.encode(list(group['option_text']))
    out = util.dot_score(query_emb, item_embs)
    scores.append(out.detach().numpy()[0])

import numpy as np
scores_df = (
    pd.DataFrame({'query': queries, 'score': scores, 'option_text': items})
    .explode(["score", "option_text"])
)

scores_df.to_csv("query_dense_scores.csv")