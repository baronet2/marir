# See https://huggingface.co/facebook/bart-large-mnli
from transformers import pipeline
classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")


sequence_to_classify = "I would like meat lasagna but I'm watching my weight"

candidate_labels = [
    'Vegetarian lasagna with mushrooms, mixed vegetables, textured vegetable protein, and meat replacement',
    'Forgot the Meat Lasagna with onions, mushrooms and spinach',
    'Beef lasagna with whole-wheat noodles, low-fat cottage cheese, and part-skim mozzarella cheese',
    'Cheesy lasagna with Italian sausage, mushrooms, and 8 types of cheese',
    'Meat loaf containing vegetables such as potatoes, onions, corn, carrots, and cabbage'
]

print(classifier(sequence_to_classify, candidate_labels))
