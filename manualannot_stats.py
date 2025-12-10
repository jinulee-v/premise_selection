import os
import json
import numpy
from matplotlib import pyplot as plt
import math

data = []
for filename in os.listdir("manual_annot/data"):
    if not filename.endswith(".json"):
        continue
    with open(os.path.join("manual_annot/data", filename), "r") as f:
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

# PRM result stats
for prm in ["versaprm", "qwenprm-7b", "gpt-5-mini", "qwen-2.5-7b-instruct"]:
    print("PRM Model:", prm)
    for filename in os.listdir("results"):
        if filename != f"manualannot_{prm}_results.jsonl":
            continue
        with open(os.path.join("results", filename), "r") as f:
            data = []
            for line in f:
                data.append(json.loads(line))
    
    # Print F1 score for predicting "correct" vs. "first error" using either loss_full_premise or loss_selected_premise
    tp_full = 0
    fp_full = 0
    fn_full = 0
    tp_selected = 0
    fp_selected = 0
    fn_selected = 0
    tp_noprem = 0
    fp_noprem = 0
    fn_noprem = 0
    for item in data:
        for step_data in item["results"]:
            if step_data["label"] is None:
                continue
            prediction_full = 1 if step_data["score_full_premise"] > 0.5 else 0
            prediction_selected = 1 if step_data["score_selected_premise"] > 0.5 else 0
            prediction_noprem = 1 if step_data["score_no_premise"] > 0.5 else 0
            if step_data["label"] == 0:
                if prediction_full == 0:
                    tp_full += 1
                else:
                    fn_full += 1
                if prediction_selected == 0:
                    tp_selected += 1
                else:
                    fn_selected += 1
                if prediction_noprem == 0:
                    tp_noprem += 1
                else:
                    fn_noprem += 1
            else:
                if prediction_full == 0:
                    fp_full += 1
                if prediction_selected == 0:
                    fp_selected += 1
                if prediction_noprem == 0:
                    fp_noprem += 1
    precision_full = tp_full / (tp_full + fp_full) if (tp_full + fp_full) > 0 else 0
    recall_full = tp_full / (tp_full + fn_full) if (tp_full + fn_full) > 0 else 0
    f1_full = 2 * precision_full * recall_full / (precision_full + recall_full) if (precision_full + recall_full) > 0 else 0
    precision_selected = tp_selected / (tp_selected + fp_selected) if (tp_selected + fp_selected) > 0 else 0
    recall_selected = tp_selected / (tp_selected + fn_selected) if (tp_selected + fn_selected) > 0 else 0
    f1_selected = 2 * precision_selected * recall_selected / (precision_selected + recall_selected) if (precision_selected + recall_selected) > 0 else 0
    precision_noprem = tp_noprem / (tp_noprem + fp_noprem) if (tp_noprem + fp_noprem) > 0 else 0
    recall_noprem = tp_noprem / (tp_noprem + fn_noprem) if (tp_noprem + fn_noprem) > 0 else 0
    f1_noprem = 2 * precision_noprem * recall_noprem / (precision_noprem + recall_noprem) if (precision_noprem + recall_noprem) > 0 else 0
    print(f"Full Premise - Precision: {precision_full:.4f}, Recall: {recall_full:.4f}, F1: {f1_full:.4f}")
    print(f"Selected Premise - Precision: {precision_selected:.4f}, Recall: {recall_selected:.4f}, F1: {f1_selected:.4f}")
    print(f"No Premise - Precision: {precision_noprem:.4f}, Recall: {recall_noprem:.4f}, F1: {f1_noprem:.4f}")

    # Calculate loss difference at step_idx == 5
    full_loss_idx5 = []
    selected_loss_idx5 = []
    noprem_loss_idx5 = []
    for item in data:
        if len(item["results"]) < 5:
            continue
        step_data = item["results"][5-1]  # step_idx is 1-based
        if step_data["label"] is None:
            continue
        loss_full_premise = step_data["loss_full_premise"]
        loss_noprem = step_data["loss_no_premise"]
        loss_selected_premise = step_data["loss_selected_premise"]
        full_loss_idx5.append(loss_full_premise)
        noprem_loss_idx5.append(loss_noprem)
        selected_loss_idx5.append(loss_selected_premise)
    
    # Calculate loss difference at first error step
    full_loss_first_error = []
    selected_loss_first_error = []
    noprem_loss_first_error = []
    for item in data:
        first_error = -1
        for step_data in item["results"]:
            if step_data["label"] == 0:
                first_error = step_data["step_idx"]
        if first_error == -1:
            continue
        step_data = item["results"][first_error-1]  # step_idx is 1-based
        if step_data["label"] is None:
            continue
        loss_full_premise = step_data["loss_full_premise"]
        loss_selected_premise = step_data["loss_selected_premise"]
        loss_noprem = step_data["loss_no_premise"]
        full_loss_first_error.append(loss_full_premise)
        selected_loss_first_error.append(loss_selected_premise)
        noprem_loss_first_error.append(loss_noprem)
    
    # Calculate loss difference at step_idx == 5
    full_loss_all = []
    selected_loss_all = []
    noprem_loss_all = []
    for item in data:
        for step_data in item["results"]:
            if step_data["label"] is None:
                continue
            loss_full_premise = step_data["loss_full_premise"]
            loss_selected_premise = step_data["loss_selected_premise"]
            loss_noprem = step_data["loss_no_premise"]
            full_loss_all.append(loss_full_premise)
            selected_loss_all.append(loss_selected_premise)
            noprem_loss_all.append(loss_noprem)

    # Plot histograms
    bins = numpy.linspace(0, math.log(2) * 10, 31)
    plt.figure(figsize=(18, 5))
    plt.subplot(1, 3, 1)
    plt.hist(full_loss_idx5, bins=bins, color='#feb253', alpha=0.7)
    plt.hist(selected_loss_idx5, bins=bins, color='#00afbd', alpha=0.7)
    plt.hist(noprem_loss_idx5, bins=bins, color='#ff8ca1', alpha=0.7)
    plt.title(f'PRM: {prm} - Loss at Step 5')
    plt.xlabel('CE Loss')
    plt.ylabel('Frequency')

    plt.subplot(1, 3, 2)
    plt.hist(full_loss_first_error, bins=bins, color='#feb253', alpha=0.7)
    plt.hist(selected_loss_first_error, bins=bins, color='#00afbd', alpha=0.7)
    plt.hist(noprem_loss_first_error, bins=bins, color='#ff8ca1', alpha=0.7)
    plt.title(f'PRM: {prm} - Loss at First Error Step')
    plt.xlabel('CE Loss')
    plt.ylabel('Frequency')
    
    plt.subplot(1, 3, 3)
    plt.hist(full_loss_all, bins=bins, color='#feb253', alpha=0.7)
    plt.hist(selected_loss_all, bins=bins, color='#00afbd', alpha=0.7)
    plt.hist(noprem_loss_all, bins=bins, color='#ff8ca1', alpha=0.7)
    plt.title(f'PRM: {prm} - Loss at All Steps')
    plt.xlabel('CE Loss')
    plt.ylabel('Frequency')

    plt.tight_layout()

    plt.legend(['Full Premises', 'Selected Premises'])

    plt.savefig("plots/manualannot_stats_loss_" + prm + ".svg")
    plt.close()

    # Entropy distribution of three settings (full, selected, no premise)
    # entropy = - [p log(p) + (1-p) log(1-p)] (in bits)
    full_entropy = []
    selected_entropy = []
    noprem_entropy = []
    for item in data:
        for step_data in item["results"]:
            if step_data["label"] is not None:
                continue
            score_full = step_data["score_full_premise"]
            score_selected = step_data["score_selected_premise"]
            score_noprem = step_data["score_no_premise"]
            entropy_full = - (score_full * math.log2(score_full) + (1 - score_full) * math.log2(1 - score_full))
            entropy_selected = - (score_selected * math.log2(score_selected) + (1 - score_selected) * math.log2(1 - score_selected))
            entropy_noprem = - (score_noprem * math.log2(score_noprem) + (1 - score_noprem) * math.log2(1 - score_noprem))
            full_entropy.append(entropy_full)
            selected_entropy.append(entropy_selected)
            noprem_entropy.append(entropy_noprem)
    # Plot histograms
    bins = numpy.linspace(0, 1, 21)
    plt.figure(figsize=(18, 5))
    plt.subplot(1, 3, 1)
    plt.hist(full_entropy, bins=bins, color='#feb253', alpha=0.7)
    plt.title(f'PRM: {prm} - Entropy with Full Premises')
    plt.xlabel('Entropy')
    plt.ylabel('Frequency')
    plt.subplot(1, 3, 2)
    plt.hist(selected_entropy, bins=bins, color='#00afbd', alpha=0.7)
    plt.title(f'PRM: {prm} - Entropy with Selected Premises')
    plt.xlabel('Entropy')
    plt.ylabel('Frequency')
    plt.subplot(1, 3, 3)
    plt.hist(noprem_entropy, bins=bins, color='#ff8ca1', alpha=0.7)
    plt.title(f'PRM: {prm} - Entropy with No Premises')
    plt.xlabel('Entropy')
    plt.ylabel('Frequency')
    # Align ylim
    max_ylim = max(plt.subplot(1, 3, 1).get_ylim()[1], plt.subplot(1, 3, 2).get_ylim()[1], plt.subplot(1, 3, 3).get_ylim()[1])
    plt.subplot(1, 3, 1).set_ylim(0, max_ylim)
    plt.subplot(1, 3, 2).set_ylim(0, max_ylim)
    plt.subplot(1, 3, 3).set_ylim(0, max_ylim)
    plt.tight_layout()
    plt.savefig("plots/manualannot_stats_entropy_" + prm + ".svg")
    plt.close()

    # Loss diff per label
    loss_diff_per_label = {"correct": [], "first_error": [], "": []}
    for item in data:
        for step_data in item["results"]:
            if step_data["label"] is None:
                label_key = ""
                continue
            elif step_data["label"] == 1:
                label_key = "correct"
            else:
                label_key = "first_error"
            loss_full_premise = step_data["loss_full_premise"]
            loss_selected_premise = step_data["loss_selected_premise"]
            loss_diff = loss_selected_premise - loss_full_premise
            loss_diff_per_label[label_key].append(loss_diff)
    # Plot histograms
    bins = numpy.linspace(-math.log(2) * 10, math.log(2) * 10, 20)
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.hist(loss_diff_per_label["correct"], bins=bins, color='#00afbd', alpha=0.7)
    plt.title(f'PRM: {prm} - Loss Difference for Correct Steps')
    plt.xlabel('Loss Selected - Loss Full')
    plt.ylabel('Frequency')
    plt.subplot(1, 2, 2)
    plt.hist(loss_diff_per_label["first_error"], bins=bins, color='#feb253', alpha=0.7)
    plt.title(f'PRM: {prm} - Loss Difference for First Error Steps')
    plt.xlabel('Loss Selected - Loss Full')
    plt.ylabel('Frequency')
    plt.tight_layout()
    plt.savefig("plots/manualannot_stats_lossdiff_" + prm + ".svg")
    plt.close()