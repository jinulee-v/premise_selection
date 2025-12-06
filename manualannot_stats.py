"""
Use OpenAI `gpt-5.1` to identify errors in the reasoning trace.
"""

import os
import json

data = []
for filename in os.listdir("manual_annot/final_data"):
    if not filename.endswith(".json"):
        continue
    with open(os.path.join("manual_annot/final_data", filename), "r") as f:
        data.append(json.load(f))

print("Loaded", len(data), "examples.")

# Count how many have first_error is -1/something else and final_answer_correct is True/False in a 2x2 table
counts = {
    (True, True): 0,
    (True, False): 0,
    (False, True): 0,
    (False, False): 0,
}
for example in data:
    first_error = example["metadata"]["first_error_index"]
    final_answer_correct = example["metadata"]["final_answer_correct"]
    has_error = first_error != -1
    counts[(has_error, final_answer_correct)] += 1
print("Counts (has_error, final_answer_correct):")
for key, value in counts.items():
    print(f"{key}: {value}")

# For both groups with has_error == True, check if the first error step connects to the final step of the reasoning
def connected(example, step_1, step_2):
    # adjacency list
    adj = {}
    for edge in example["edges"]:
        if edge["from_node_id"] not in adj:
            adj[edge["from_node_id"]] = []
        adj[edge["from_node_id"]].append(edge["to_node_id"])
    visited = set()
    def dfs(node_id):
        if node_id in visited:
            return
        visited.add(node_id)
        for neighbor in adj.get(node_id, []):
            dfs(neighbor)
    dfs(step_1) # floodfill
    return step_2 in visited

# Group 1: has_error == True and final_answer_correct == False
count_connected_1 = 0
total_1 = 0
for example in data:
    first_error = example["metadata"]["first_error_index"]
    final_answer_correct = example["metadata"]["final_answer_correct"]
    has_error = first_error != -1
    if has_error and not final_answer_correct:
        total_1 += 1
        step_1 = example["nodes"][first_error+1]["id"]
        assert step_1 == "trace" + str(first_error)
        step_2 = example["nodes"][-1]["id"]
        if connected(example, step_1, step_2):
            count_connected_1 += 1
print(f"Group 1 (has_error == True and final_answer_correct == False): {count_connected_1}/{total_1} connected ({(count_connected_1/total_1*100) if total_1 > 0 else 0:.2f}%)")

# Group 2: has_error == True and final_answer_correct == True
count_connected_2 = 0
total_2 = 0
for example in data:
    first_error = example["metadata"]["first_error_index"]
    final_answer_correct = example["metadata"]["final_answer_correct"]
    has_error = first_error != -1
    if has_error and final_answer_correct:
        total_2 += 1
        step_1 = example["nodes"][first_error+1]["id"]
        assert step_1 == "trace" + str(first_error)
        step_2 = example["nodes"][-1]["id"]
        if connected(example, step_1, step_2):
            count_connected_2 += 1
print(f"Group 2 (has_error == True and final_answer_correct == True): {count_connected_2}/{total_2} connected ({(count_connected_2/total_2*100) if total_2 > 0 else 0:.2f}%)")
