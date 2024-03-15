We have kept the following queries:

```
multi_aspect_queries = {
    'Italian place with a burger': ['Italian place', 'burger'],
    'A cafe that also offers beer': ['cafe', 'beer'],
    'Japanese restaurant with pasta': ['Japanese restaurant', 'pasta'],
    'An ice cream shop with bubble tea': ['ice cream shop', 'bubble tea'],
    'I am in search of a fancy Pakistani restaurant with authentic food': ['fancy', 'Pakistani restaurant']
}
```

For each of these queries:
1. There is only one restaurant marked as relevant
2. There are clearly two aspects that must be satisfied together

We have kept only the first 5 reviews per restaurant.