import argparse
from tqdm import tqdm
import json
import math

from premise_selection.dataset import load_premise_selection_dataset, all_step_combinations
from premise_selection.prm import PRM_dict

def main(args):
    dataset = load_premise_selection_dataset(args.dataset_path)
    # DEBUG
    # dataset = dataset.select(range(10))
    prm_class = PRM_dict[args.prm_model]
    prm = prm_class(device="cuda")

    results = {} # id -> result
    for item in tqdm(dataset):
        question = item['question']
        steps = [question] + item['steps']
        # assert steps[0] == question
        labels = [None] + item['labels']
        if len([l for l in labels if l is not None]) < 5:
            # Need at least one positive and one negative label
            continue
    
        steps = steps[:6]  # question + first 5 steps
        labels = labels[:6]

        # Get all step combinations
        indices = all_step_combinations(steps)
        step_combinations = []
        for idx_tuple in indices:
            comb = [steps[i] for i in idx_tuple]
            step_combinations.append(comb)

        # Get scores from PRM
        scores = prm.get_combination_scores(step_combinations)
        assert len(scores) == len(step_combinations) and len(scores) == len(indices)

        # Store results
        result = []
        for idx_tuple, score in zip(indices, scores):
            # score: [-inf, 0] (log prob of being correct)
            label = labels[idx_tuple[-1]] # 0 or 1
            # log cross entropy
            loss = - (label * score + (1 - label) * math.log(1 - math.exp(score) + 1e-12))
            result.append({
                'step_indices': idx_tuple,
                'score': math.exp(score),
                'label': label,
                'loss': loss
            })
        results[item['id']] = result
    
    # Save results
    output_path = f'results/{args.dataset}_{args.prm_model}_results.jsonl'
    with open(output_path, 'w') as f:
        for id, res in results.items():
            f.write(json.dumps({'id': id, 'results': res}) + '\n')

if __name__ == "__main__":
    dataset_list = ['processbench', 'mrben']  # Add more datasets as needed
    parser = argparse.ArgumentParser()
    parser.add_argument('--prm_model', type=str, default='versaprm', choices=list(PRM_dict.keys()), help='PRM model to use')
    parser.add_argument('--dataset', type=str, default='processbench', choices=dataset_list, help='Path to the dataset file')
    args = parser.parse_args()
    
    # add dataset_path argument
    setattr(args, 'dataset_path', f'data/{args.dataset}.jsonl')

    main(args)
