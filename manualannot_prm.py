import argparse
from tqdm import tqdm
import json
import math
import os

from premise_selection.prm import PRM_dict

def main(args):
    dataset = []
    for filename in os.listdir("manual_annot/final_data"):
        if not filename.endswith(".json"):
            continue
        with open(os.path.join("manual_annot/final_data", filename), "r") as f:
            dataset.append(json.load(f))
    print("Loaded", len(dataset), "examples.")

    # DEBUG
    # dataset = dataset.select(range(10))
    prm_class = PRM_dict[args.prm_model]
    prm = prm_class(device="cuda")

    results = {} # id -> result
    for item in tqdm(dataset):
        question = item['nodes'][0]['text']
        steps = [question] + [node['text'] for node in item['nodes'][1:]]
        # assert steps[0] == question
        label_map = {"correct": 1, "first_error": 0, "": None}
        labels = [None] + [label_map[node.get('label', "")] for node in item['nodes'][1:]]
        if len([l for l in labels if l is not None]) < 5:
            # Need at least one positive and one negative label
            continue

        # Use full steps
        step_combinations_full = []
        for index in range(1, len(steps)):
            step_combinations_full.append(steps[:index])
        step_combinations_selected = []
        for index in range(1, len(steps)):
            # Find premises
            from_ids = []
            for edge in item['edges']:
                if edge['to_node_id'] == item['nodes'][index]['id']:
                    from_ids.append(int(edge['from_node_id'].replace("trace", "")) + 1)
            selected_steps = []
            for node in item['nodes']:
                if node['id'] in from_ids:
                    selected_steps.append(node['text'])
            step_combinations_selected.append(selected_steps)
        
        assert len(step_combinations_full) == len(step_combinations_selected)
        # Combine both full and selected steps
        step_combinations = step_combinations_full + step_combinations_selected

        # Get scores from PRM
        scores = prm.get_combination_scores(step_combinations)
        assert len(scores) == len(step_combinations)
        
        # split back in half
        scores_full = scores[:len(step_combinations_full)]
        scores_selected = scores[len(step_combinations_full):]

        # Store results
        result = []
        for step_idx, fullscore, selectedscore in zip(range(1, len(steps)), scores_full, scores_selected):
            # score: [-inf, 0] (log prob of being correct)
            label = labels[step_idx] # 0, 1, or None
            # log cross entropy
            if label is not None:
                loss_full = - (label * fullscore + (1 - label) * math.log(1 - math.exp(fullscore) + 1e-12))
                loss_selected = - (label * selectedscore + (1 - label) * math.log(1 - math.exp(selectedscore) + 1e-12))
            else:
                loss_full = None
                loss_selected = None
            result.append({
                'step_idx': step_idx,
                'score_full_premise': math.exp(fullscore),
                'loss_full_premise': loss_full,
                'score_selected_premise': math.exp(selectedscore),
                'loss_selected_premise': loss_selected,
                'label': label,
            })
        results[item['doc_id']] = result
    
    # Save results
    output_path = f'results/manualannot_{args.prm_model}_results.jsonl'
    with open(output_path, 'w') as f:
        for id, res in results.items():
            f.write(json.dumps({'id': id, 'results': res}) + '\n')

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--prm_model', type=str, default='versaprm', choices=list(PRM_dict.keys()), help='PRM model to use')
    args = parser.parse_args()
    
    # add dataset_path argument
    setattr(args, 'dataset_path', f'data/{args.dataset}.jsonl')

    main(args)
