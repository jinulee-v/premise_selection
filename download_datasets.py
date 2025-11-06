import json
from datasets import load_dataset

# Data format:
"""
[
    {
        "id": "unique_id",
        "metadata": {} # optional
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

# ProcessBench
print("Downloading and processing ProcessBench...")
processbench = []
splits = ["gsm8k", "math", "olympiadbench", "omnimath"]
for split in splits:
    dataset = load_dataset("Qwen/ProcessBench", split=split)
    for item in dataset:
        label = int(item["label"])
        if label == -1:
            # all steps are correct
            processbench.append({
                "id": "processbench-" + item["id"],
                "metadata": {
                    "subject": "math",
                    "generator": item["generator"],
                    "final_answer_correct": item["final_answer_correct"]
                },
                "question": item["problem"],
                "steps": item["steps"],
                "labels": [1] * len(item["steps"])
            })
        else:
            # 0~label (inclusive) are correct steps
            steps = item["steps"] # [:label + 1]
            labels = [1] * (label) + [0] + [None] * (len(steps) - label - 1)
            assert len(steps) == len(labels)
            processbench.append({
                "id": "processbench-" + item["id"],
                "metadata": {
                    "subject": "math",
                    "generator": item["generator"],
                    "final_answer_correct": item["final_answer_correct"]
                },
                "question": item["problem"],
                "steps": steps,
                "labels": labels
            })
# Store as JSONL
with open("data/processbench.jsonl", "w") as f:
    for item in processbench:
        f.write(json.dumps(item) + "\n")
print("ProcessBench dataset saved to data/processbench.jsonl", len(processbench), "examples")

# MR-Ben
print("Downloading and processing MR-Ben...")
mrben = []
dataset = load_dataset("Randolphzeng/Mr-Ben", split="train")
for i, item in enumerate(dataset):
    if item["Model_Solution_First_Error_Step"] == "N/A":
        mrben.append({
            "id": "mrben-" + str(i),
            "metadata": {
                "subject": item["Subject"],
                "generator": item["Sampled_Model"],
                "final_answer_correct": item["Model_Solution_Correctness"] == "correct"
            },
            "question": item["Question"] + "\n" + item["Options"],
            "steps": item["Model_Solution_Steps"],
            "labels": [1] * len(item["Model_Solution_Steps"])
        })
    else:
        try:
            label = int(item["Model_Solution_First_Error_Step"])-1 # MR-Ben has 1-based indexing
            # 0~label (inclusive) are correct steps
            steps = item["Model_Solution_Steps"][:label + 1]
            labels = [1] * (label) + [0] + [None] * (len(steps) - label - 1)
            assert len(steps) == len(labels), f"Length mismatch in MR-Ben processing: {len(steps)} steps vs {len(labels)} labels"
            mrben.append({
                "id": "mrben-" + str(i),
                "metadata": {
                    "subject": item["Subject"],
                    "generator": item["Sampled_Model"],
                    "final_answer_correct": item["Model_Solution_Correctness"] == "correct"
                },
                "question": item["Question"] + "\n" + item["Options"],
                "steps": steps,
                "labels": labels
            })
        except ValueError:
            print(f"Skipping MR-Ben item {i} due to invalid First_Error_Step: {item['Model_Solution_First_Error_Step']}")
        except Exception as e:
            print(f"Error processing MR-Ben item {i}: {e}")

with open("data/mrben.jsonl", "w") as f:
    for item in mrben:
        f.write(json.dumps(item) + "\n")
print("MR-Ben dataset saved to data/mrben.jsonl:", len(mrben), "examples")

# VersaPRM training data
print("Downloading and processing VersaPRM training data...")
versaprm = []
dataset = load_dataset("UW-Madison-Lee-Lab/MMLU-Pro-CoT-Train-Labeled", split="train")
for i, item in enumerate(dataset):
    # Convert first -1 to 0
    labels = eval(item["labels"])
    for idx, label in enumerate(labels):
        if label == -1:
            labels[idx] = 0
            break
    versaprm.append({
        "id": "versaprm-" + item["id"] + "-cot" + str(item["cot_id"]),
        "metadata": {
            "subject": item["category"],
            "source": item["src"],
            "generator": None,
            "final_answer_correct": item["parsed_answer_correctness"]
        },
        "question": item["question"],
        "steps": eval(item["chain_of_thoughts"]),
        "labels": labels
    })

with open("data/versaprm.jsonl", "w") as f:
    for item in versaprm:
        f.write(json.dumps(item) + "\n")
print("VersaPRM dataset saved to data/versaprm.jsonl:", len(versaprm), "examples")