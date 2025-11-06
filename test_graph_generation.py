from vllm import LLM, SamplingParams
from vllm.sampling_params import GuidedDecodingParams
import json
from typing import Dict, List
from pydantic import BaseModel

# Instantiate the vLLM model
llm = LLM(model="Qwen/Qwen3-4B-Instruct-2507")

class EntailmentGraph(BaseModel):
    graph: Dict[str, List[int]]

def generate_entailment_graph(question: str, steps: list[str]) -> Dict[int, List[int]]:
    """
    Use Qwen model via vLLM to generate a coherent DAG entailment graph.
    Each node (step) lists its premises as node indices (integers).

    Args:
        question (str): The main question (node 0).
        steps (list[str]): List of reasoning steps (nodes 1..N).

    Returns:
        dict[int, list[int]]: Adjacency dictionary mapping node_id -> list of premise node_ids.
    """

    # Construct the reasoning context
    context = f"Step 0 (Question): {question}\n"
    for i, step in enumerate(steps, start=1):
        context += f"Step {i}: {step}\n"

    # Define the prompt
    prompt = (
        "You are given a reasoning process consisting of a question and reasoning steps.\n"
        "Construct a *coherent entailment DAG* where each node (step) depends on zero or more previous nodes (premises).\n"
        "Premises should provide all necessary information to judge if the step is correct, and track error propagation.\n"
        "Return a JSON object mapping each node ID to a list of its premise node IDs, starting from 0.\n"
        "Ensure the graph is acyclic and all premise IDs are less than the node's ID.\n\n"
        "Example format:\n"
        "Step 0 (Question): ...\n"
        "...\n"
        "Step 7: ...\n\n"
        "{\n"
        "  \"graph\": {\n"
        "    \"0\": [],\n"
        "    \"1\": [0],\n"
        "    ...\n"
        "    \"7\": [4, 6]\n"
        "  }\n"
        "}\n"
        "Process:\n\n"
        f"{context}\n"
        "Now output the entailment graph as JSON object. Only output the JSON and nothing else."
    )
    print(prompt)

    # Define JSON-constrained sampling
    sampling_params = SamplingParams(
        temperature=1.0,
        max_tokens=512,
        guided_decoding=GuidedDecodingParams(
            json=EntailmentGraph.model_json_schema()
        )
    )

    # Run model inference
    outputs = llm.generate(
        [prompt],
        sampling_params=sampling_params
    )
    text_output = outputs[0].outputs[0].text
    text_output = text_output.split("{", 1)[-1].rsplit("}", 1)[0] # First { and last }
    text_output = "{" + text_output.strip() + "}"

    # Parse JSON safely
    try:
        graph = json.loads(text_output)
        # convert keys to int
        graph["graph"] = {int(k): v for k, v in graph["graph"].items()}
    except Exception as e:
        raise ValueError(f"Model output was not valid JSON: {text_output}") from e

    return graph

if __name__ == "__main__":
    question = "Jason was told he could earn $3.00 for doing his laundry,  $1.50 for cleaning his room, $0.75 for taking the trash to the curb each week and $0.50 for emptying the dishwasher.  In a two week period, Jason emptied the dishwasher 6 times, did his laundry once, took the trash out twice and cleaned his room once.  How much money did Jason earn?"
    steps = [
        "To find out how much money Jason earned in a two-week period, we need to calculate the total earnings from each task and then add them together.",
        "First, calculate the earnings from emptying the dishwasher. The rate for emptying the dishwasher is $0.50 per time. Jason emptied the dishwasher 6 times in two weeks, so the total earnings from this task is: 6 * $0.50 = $3.00.",
        "Second, calculate the earnings from doing laundry. The rate for doing laundry is $3.00 per time. Jason did his laundry once, so the total earnings from this task is: 1 * $3.00 = $3.00.",
        "Third, calculate the earnings from taking the trash out. The rate for taking the trash out is $0.75 per time. Jason took the trash out twice, so the total earnings from this task is: 2 * $0.75 = $1.50.",
        "Fourth, calculate the earnings from cleaning his room. The rate for cleaning his room is $1.50 per time. Jason cleaned his room once, so the total earnings from this task is: 1 * $1.50 = $1.50.",
        "Finally, add up the earnings from all tasks to get the total amount Jason earned. Total earnings = $3.00 (dishwasher) + $3.00 (laundry) + $1.50 (trash) + $1.50 (room). Total earnings = $8.00.",
        "So, Jason earned a total of $\\boxed{$8.00}$ in a two-week period."
    ]
    label = [1, 1, 1, 1, 1, 0, None]
    final_answer_correct = False

    graphs = []
    for i in range(5):
        graph = generate_entailment_graph(question, steps)
        graphs.append(graph)
    for i, g in enumerate(graphs):
        print(f"Graph {i+1}: {g}")