import random
import json

list = []

url = 'ipfs://QmPVK74FuuVpWQDr8xwWRrgwDe6a5dqWW2CXuLjKTPUNDb'

for i in range(0, 10000):
    list.append({
        'url': url,
        'position': [
            random.uniform(-5.0, 5.0),
            random.uniform(0.0, 1.0),
            random.uniform(-5.0, 5.0),
        ],
        'scale': 0.25,
        'rotation': [
            random.uniform(0.0, 360.0),
            random.uniform(0.0, 360.0),
            random.uniform(0.0, 360.0),
        ]
    })

print(json.dumps(list))