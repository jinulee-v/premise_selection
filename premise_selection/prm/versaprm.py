import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

class VersaPRM:
    def __init__(self, device="cuda", batch_size=16, **kwargs):
        self.device = device
        self.tokenizer = self._get_tokenizer('UW-Madison-Lee-Lab/VersaPRM')
        self.model = AutoModelForCausalLM.from_pretrained('UW-Madison-Lee-Lab/VersaPRM')
        self.model.to(self.device)
        self.batch_size = batch_size

    def _get_tokenizer(self, model_id):
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        tokenizer.pad_token = tokenizer.eos_token  
        tokenizer.padding_side = 'left' 
        tokenizer.truncation_side = 'left'
        return tokenizer

    def get_combination_scores(self, step_combinations):
        input_texts = []
        for comb in step_combinations:
            if 0 in comb:
                # question is included
                input_texts.append(comb[0] + ' \n\n' + ' \n\n\n\n'.join(comb[1:]) + ' \n\n\n\n')
            else:
                # question is not included
                input_texts.append(' \n\n\n\n'.join(comb) + ' \n\n\n\n')
        input_ids = self.tokenizer(input_texts, padding=True, truncation=True, return_tensors="pt").input_ids.to(self.device)

        step_predictions = []
        for start in range(0, len(input_ids), self.batch_size):
            end = min(start + self.batch_size, len(input_ids))
            input_id = input_ids[start:end]
            with torch.no_grad():
                logits = self.model(input_id).logits[:, :, [12, 10]] # (batch_size, seq_len, vocab_size)
                scores = logits.log_softmax(dim=-1)
                step_scores = scores[:, -1, 1]
                step_predictions.extend(step_scores.tolist())
        
        assert len(step_predictions) == len(step_combinations)
        return step_predictions

if __name__ == "__main__":
    prm = VersaPRM(device="cuda")
    question = 'Question: In Python 3, which of the following function convert a string to an int in python?\nA. short(x)\nB. float(x)\nC. integer(x [,base])\nD. double(x)\nE. int(x [,base])\nF. long(x [,base] )\nG. num(x)\nH. str(x)\nI. char(x)\nJ. digit(x [,base])'
    steps = ["To convert a string to an integer in Python 3, we use the built-in function int().",
             "The int() function takes two arguments: the string to be converted and an optional base (default is 10, which is for decimal).",
             "For example: int(\"123\", 10) converts the string \"123\" to the integer 123.",
             "Looking at the options, we can see that the correct function is option E: int(x [,base]).",
             "The answer is (E)."]