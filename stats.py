from premise_selection.prm import PRM_dict
import json
from collections import defaultdict
import matplotlib.pyplot as plt
import numpy as np
import os
import math

data_fifth = {} # model -> dataset -> list of (premise, label, loss)

for model in PRM_dict:
    data_fifth[model] = {}
    for dataset in ["processbench", "mrben"]:
        data_fifth[model][dataset] = []
        if os.path.exists(f'results/{dataset}_{model}_results.jsonl') == False:
            print(f'File results/{dataset}_{model}_results.jsonl does not exist, skipping.')
            continue
        with open(f'results/{dataset}_{model}_results.jsonl', 'r') as f:
            for i, line in enumerate(f):
                data_fifth[model][dataset].append([])
                item = json.loads(line)
                for res in item['results']:
                    premise = res['step_indices']
                    label = res['label']
                    loss = res['loss']
                    score = res['score']
                    if premise[-1] == 5:
                        data_fifth[model][dataset][i].append((premise, label, loss, score))

        # 1. Plot the loss per length of premise set
        losses = defaultdict(list)
        for i in range(len(data_fifth[model][dataset])):
            full_set_loss = float('inf')
            for premise, label, loss, score in data_fifth[model][dataset][i]:
                if premise == list(range(6)):
                    full_set_loss = loss
                    break
            if full_set_loss == float('inf'):
                continue

            min_per_length = {x: float('inf') for x in range(6)}
            for premise, label, loss, score in data_fifth[model][dataset][i]:
                losses[len(premise) - 1].append(loss)
        # Boxplot for losses
        lengths = sorted(losses.keys())
        data = [losses[length] for length in lengths]
        plt.figure(figsize=(10, 6))
        plt.boxplot(data, labels=lengths)
        plt.xlabel('Length of Premise')
        plt.ylabel('Loss(premise set)')
        plt.title(f'Loss by Premise Length for Model {model} on {dataset}')
        plt.savefig(f'plots/loss_by_premise_length_{model}_{dataset}.svg')
        plt.close()

        # 2. Count the premise set that yields the minimum loss and aggregate by premise set size
        min_premises = defaultdict(int)
        for i in range(len(data_fifth[model][dataset])):
            min_loss = float('inf')
            min_premise = None
            for premise, label, loss, score in data_fifth[model][dataset][i]:
                if loss < min_loss:
                    min_loss = loss
                    min_premise = premise
            # Count `premise` that has minimum loss
            min_premises[tuple(min_premise)] += 1
        print(f'Model: {model}, Dataset: {dataset}')
        for premise, count in sorted(min_premises.items(), key=lambda x: x[1], reverse=True):
            print(f'  Premise: {premise}, Count: {count}')
        # Sum everything else than (0,1,2,3,4,5)
        other_count = 0
        for premise, count in min_premises.items():
            if premise != (0,1,2,3,4,5):
                other_count += count
        print(f'  Other premises count: {other_count}')

        # Generate boxplot for min loss for each len(premise)
        length_to_losses = defaultdict(list)
        for i in range(len(data_fifth[model][dataset])):
            min_loss_per_length = {}
            for premise, label, loss, score in data_fifth[model][dataset][i]:
                length = len(premise) - 1  # exclude current step
                if length not in min_loss_per_length or loss < min_loss_per_length[length]:
                    min_loss_per_length[length] = loss
            for length, loss in min_loss_per_length.items():
                length_to_losses[length].append(loss)

        # Plotting
        lengths = sorted(length_to_losses.keys())
        data = [length_to_losses[length] for length in lengths]

        plt.figure(figsize=(10, 6))
        plt.boxplot(data, labels=lengths)
        plt.xlabel('Length of Premise')
        plt.ylabel('Minimum Loss')
        plt.title(f'Minimum Loss by Premise Length for Model {model} on {dataset}')
        plt.savefig(f'plots/min_loss_by_premise_length_{model}_{dataset}.svg')
        plt.close()        

        # Plot max F1 for any premise set and plot with full premise
        full_correct = 0
        any_correct = 0
        total = 0
        for i in range(len(data_fifth[model][dataset])):
            total += 1
            any_flag = False
            full_flag = False
            for premise, label, loss, score in data_fifth[model][dataset][i]:
                if (score >= 0.5 and label == 1) or (score < 0.5 and label == 0):
                    any_flag = True
                    if list(premise) == list(range(6)):
                        full_flag = True
            if full_flag:
                full_correct += 1
            if any_flag:
                any_correct += 1
                    
                
        print(f'Model: {model}, Dataset: {dataset}')
        print(f'  Full premise set F1: {full_correct / total:.4f}')
        print(f'  Any premise set F1: {any_correct / total:.4f}')
        plt.figure(figsize=(6, 6))
        plt.bar(['Full Premise', 'Any Premise'], [full_correct / total, any_correct / total], color=['blue', 'orange'])
        plt.ylim(0, 1)
        plt.ylabel('Accuracy')
        plt.title(f'Accuracy Comparison for Model {model} on {dataset}')
        plt.savefig(f'plots/acc_comparison_{model}_{dataset}.svg')
