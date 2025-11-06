import json
from datasets import Dataset
from itertools import combinations

"""
[
    {
        "question": "problem text",
        "steps": [
            "step 1 text",
            "step 2 text",
            ...
        ],
        "labels": [0, 1, ...] # 1 for correct step, 0 for incorrect
    },
    ...
]
"""

def load_premise_selection_dataset(file_path):
    data = []
    with open(file_path, 'r') as f:
        for line in f:
            data.append(json.loads(line))
    examples = []
    for i, item in enumerate(data):
        id = item.get("id", i)
        question = item['question']
        steps = item['steps']
        labels = item['labels']
        examples.append({
            'id': id,
            'question': question,
            'steps': steps,
            'labels': labels
        })
    return Dataset.from_list(examples)

def all_step_combinations(steps):
    # needed for mapping with labels
    indices = [combo for r in range(1, len(steps) + 1) for combo in combinations(range(len(steps)), r)]
    indices = [idx for idx in indices if idx != (0,)]  # exclude the combination with only the question
    return indices