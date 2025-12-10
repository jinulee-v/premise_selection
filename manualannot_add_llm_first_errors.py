"""
Use OpenAI `gpt-5.1` to identify errors in the reasoning trace.
"""

import os
import openai
from typing import List
import json

from dotenv import load_dotenv
load_dotenv()

def identify_errors_in_trace(question: str, steps: List[str]) -> str:
    prompt = f"""
You are an expert in logical reasoning and error detection. Given a question and a series of reasoning steps, identify any errors in the reasoning process.
Question: {question}
Reasoning Steps:
"""
    for i, step in enumerate(steps, 0):
        prompt += f"[{i}] {step}\n"
    prompt += "\nIdentify any errors in the reasoning steps above, and print the index and the reason why it is wrong in JSON: {\"first_error\": <index>, \"reason\": <reason>}. If there are no errors, return -1 and leave reason empty string."

    client = openai.OpenAI(
        api_key=os.getenv("OPENAI_API_KEY")
    )
    response = client.chat.completions.create(
        model="gpt-5.1",
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.0,
        # JSON decoding
        response_format={"type": "json_object"}
    )
    # return the index
    result = json.loads(response.choices[0].message.content)
    return result["first_error"], result["reason"]

# test = identify_errors_in_trace(
#     question="What is the sum of the first ten prime numbers?",
#     steps=[
#         "The first ten prime numbers are 2, 3, 5, 7, 9, 11, 13, 17, 19, and 23.",
#         "Adding these together: 2 + 3 + 5 + 7 + 9 + 11 + 13 + 17 + 19 + 23 = 129.",
#         "Therefore, the sum of the first ten prime numbers is 129."
#     ]
# )
# print(test)

data = []
for filename in os.listdir("manual_annot/data"):
    if not filename.endswith(".json"):
        continue
    with open(os.path.join("manual_annot/data", filename), "r") as f:
        data.append(json.load(f))

print("Loaded", len(data), "examples.")
for example in data:
    question = example["nodes"][0]["text"]
    steps = [step["text"] for step in example["nodes"][1:]]
    try:
        first_error, reason = identify_errors_in_trace(question, steps)
        example["metadata"]["first_error_index"] = first_error
        example["metadata"]["first_error_reason"] = reason
        if first_error >= 0:
            for i, step in enumerate(example["nodes"]):
                if i == 0:
                    label = "question"
                elif i == first_error + 1:
                    label = "first_error"
                elif i < first_error + 1:
                    label = "correct"
                else:
                    label = ""
                step["label"] = label
        with open(os.path.join("manual_annot/data", example["doc_id"] + ".json"), "w") as f:
            json.dump(example, f, indent=4)
    except Exception as e:
        print(example["doc_id"] + " failed with error:", e)