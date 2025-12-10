import torch
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer

def make_step_rewards(logits, token_masks):
    probabilities = F.log_softmax(logits, dim=-1)
    probabilities = probabilities * token_masks.unsqueeze(-1) # bs, seq_len, num_labels
    
    all_scores_res = []
    for i in range(probabilities.size(0)):
        sample = probabilities[i] # seq_len, num_labels
        positive_probs = sample[sample != 0].view(-1, 2)[:, 1] # valid_tokens, num_labels
        non_zero_elements_list = positive_probs.cpu().tolist()
        all_scores_res.append(non_zero_elements_list)
    return all_scores_res

SYSTEM_PROMPT = "Please reason step by step, and put your final answer within \\boxed{}."

class QwenMathPRM:
    def __init__(self, device="cuda", batch_size=32, model="Qwen/Qwen2.5-Math-PRM-7B", **kwargs):
        self.device = device
        self.tokenizer = AutoTokenizer.from_pretrained(model, trust_remote_code=True)
        self.model = AutoModel.from_pretrained(model, device_map=device, trust_remote_code=True)
        self.batch_size = batch_size
        self.step_sep_id = self.tokenizer.encode("<extra_0>")[0]

    def get_combination_scores(self, step_combinations):
        input_ids = []
        for comb in step_combinations:
            if 0 in comb:
                # question is included
                messages = [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": comb[0]},
                    {"role": "assistant", "content": "<extra_0>".join(comb[1:]) + "<extra_0>"},
                ]
            else:
                # question is not included
                messages = [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": "."},
                    {"role": "assistant", "content": "<extra_0>".join(comb) + "<extra_0>"},
                ]
            conversation_str = self.tokenizer.apply_chat_template(
                messages, 
                tokenize=False, 
                add_generation_prompt=False
            )
            input_id = self.tokenizer.encode(conversation_str, return_tensors="pt").squeeze(0).to(self.device)
            input_ids.append(input_id)
        input_ids = torch.nn.utils.rnn.pad_sequence(input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id)

        step_predictions = []
        for start in range(0, len(input_ids), self.batch_size):
            end = min(start + self.batch_size, len(input_ids))
            input_id = input_ids[start:end, :]
            token_masks = (input_id == self.step_sep_id)
            with torch.no_grad():
                outputs = self.model(input_ids=input_id)
                step_reward = make_step_rewards(outputs[0], token_masks)
                step_predictions.extend([rewards[-1] for rewards in step_reward])
        
        assert len(step_predictions) == len(step_combinations)
        return step_predictions

class QwenMath7BPRM(QwenMathPRM):
    def __init__(self, device="cuda", batch_size=32, model="Qwen/Qwen2.5-Math-PRM-7B", **kwargs):
        super().__init__(device=device, batch_size=batch_size, model=model, **kwargs)
class QwenMath72BPRM(QwenMathPRM):
    def __init__(self, device="cuda", batch_size=8, model="Qwen/Qwen2.5-Math-PRM-72B", **kwargs):
        super().__init__(device=device, batch_size=batch_size, model=model, **kwargs)

if __name__ == "__main__":
    prm = QwenMathPRM(device="cuda")
    question = 'Question: In Python 3, which of the following function convert a string to an int in python?\nA. short(x)\nB. float(x)\nC. integer(x [,base])\nD. double(x)\nE. int(x [,base])\nF. long(x [,base] )\nG. num(x)\nH. str(x)\nI. char(x)\nJ. digit(x [,base])'
    steps = ["To convert a string to an integer in Python 3, we use the built-in function int().",
             "The int() function takes two arguments: the string to be converted and an optional base (default is 10, which is for decimal).",
             "For example: int(\"123\", 10) converts the string \"123\" to the integer 123.",
             "Looking at the options, we can see that the correct function is option E: int(x [,base]).",
             "The answer is (E)."]
    step_combinations = [
        [question, steps[0], steps[1], steps[2], steps[3], steps[4]],
        [steps[0], steps[3], steps[4]],
        [question, steps[1], steps[4]],
        [steps[3], steps[4]],
    ]
    scores = prm.get_combination_scores(step_combinations)
    for comb, score in zip(step_combinations, scores):
        print(f'Score: {score:.2f}')