import torch
import torch.nn.functional as F
import openai
import os
import asyncio
import math
from dotenv import load_dotenv
load_dotenv()

SYSTEM_PROMPT = """
You are a judge that evaluates the quality of reasoning steps provided to answer a question.
Assuming that all the provided context is correct, evaluate if the last step marked with \"###\" is correct.
First, analyze if the last step marked with \"###\" contains any errors. Then, output a label of \\boxed{1} (correct) or \\boxed{0} (incorrect).
"""

class OpenAIPRM:
    def __init__(self, openai_api_key=None, model="gpt-5-mini", **kwargs):
        self.openai_api_key = openai_api_key or os.getenv("OPENAI_API_KEY")
        self.client = openai.AsyncOpenAI(api_key=self.openai_api_key)
        self.model = model

    def get_combination_scores(self, step_combinations):

        async def score_combination(comb):
            prompt = SYSTEM_PROMPT + "\n\n\n"
            for i, step in enumerate(comb):
                if i == len(comb) - 1:
                    prompt += "### " + step + "\n\n"
                else:
                    prompt += step + "\n\n"
            prompt += (
                "First, analyze if the last step marked with \"###\" "
                "contains any errors. Then, output a label of \\boxed{1} (correct) "
                "or \\boxed{0} (incorrect)."
            )

            messages = [{"role": "system", "content": prompt}]

            response = await self.client.chat.completions.create(
                model=self.model,
                messages=messages,
            )

            content = response.choices[0].message.content

            if "\\boxed{1}" in content:
                return math.log(0.99)
            elif "\\boxed{0.5}" in content:
                return math.log(0.5)  # log prob of 0.5
            else:
                return math.log(0.01)

        async def runner():
            tasks = [score_combination(comb) for comb in step_combinations]
            return await asyncio.gather(*tasks)

        # If already inside a running event loop
        try:
            loop = asyncio.get_running_loop()
            # Schedule task and block synchronously until it finishes
            return loop.create_task(runner())
        except RuntimeError:
            # No event loop: safe to run normally
            return asyncio.run(runner())

class GPT5MiniPRM(OpenAIPRM):
    def __init__(self, device=None, openai_api_key=None):
        super().__init__(device=device, openai_api_key=openai_api_key, model="gpt-5-mini")

if __name__ == "__main__":
    prm = GPT5MiniPRM()
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
        [steps[4]],
    ]
    scores = prm.get_combination_scores(step_combinations)
    for comb, score in zip(step_combinations, scores):
        print(f'Score: {score:.2f}')