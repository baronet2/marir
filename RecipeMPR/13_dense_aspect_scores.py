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
    [['query', 'query_aspects', 'option_id', 'option_text']]
)

from sentence_transformers import SentenceTransformer, util
model = SentenceTransformer('sentence-transformers/msmarco-distilbert-base-tas-b')

aspects = []
items = []
scores = []
for name, group in recipe_mpr_clean.groupby('query'):
    aspects.append(group["query_aspects"].iloc[0])
    items.append([list(group['option_text'])] * len(group["query_aspects"].iloc[0]))
    aspect_embs = model.encode(group["query_aspects"].iloc[0])
    item_embs = model.encode(list(group['option_text']))
    out = util.dot_score(aspect_embs, item_embs)
    scores.append(out.detach().numpy())

scores_df = (
    pd.DataFrame({'aspect': aspects, 'score': scores, 'option_text': items})
    .explode(["aspect", "score", "option_text"])
    .explode(["score", "option_text"])
)

scores_df.to_csv("aspect_dense_scores.csv")