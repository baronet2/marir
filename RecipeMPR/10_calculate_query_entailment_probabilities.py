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

# See https://huggingface.co/facebook/bart-large-mnli
from transformers import pipeline
classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")


import json
entailment_probabilities = []

for i, (name, group) in enumerate(recipe_mpr_clean.groupby('query')):
  # Hugging face doesn't work with labels that contain commas
  query = group['query'].iloc[0].replace(",", "")
  probabilities = classifier(list(group['option_text'].values), query, multi_label=True)
  entailment_probabilities.extend(probabilities)
  if i%10 == 0:
    print(f"Saving results up to query {i}")
    with open("query_entailment_probabilities.json", "w") as f:
      json.dump(entailment_probabilities, f)

with open("query_entailment_probabilities.json", "w") as f:
  json.dump(entailment_probabilities, f)