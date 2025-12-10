from .versaprm import VersaPRM
from .qwenprm import QwenMath7BPRM, QwenMath72BPRM
from .llm_judge_openai import GPT5MiniPRM
from .llm_judge_vllm import Qwen25_7bPRM

PRM_dict = {
    "versaprm": VersaPRM,
    "qwenprm-7b": QwenMath7BPRM,
    "gpt-5-mini": GPT5MiniPRM,
    "qwen-2.5-7b-instruct": Qwen25_7bPRM,
}