# Load CodeLlama https://huggingface.co/codellama/CodeLlama-7b-Python-hf
# https://huggingface.co/meta-llama/Llama-3.2-1B?inference_api=true
import os

import requests
import tqdm
from openai import OpenAI

API_URL = "https://api-inference.huggingface.co/models/meta-llama/Llama-3.2-1B"
# Define env variable HUGGINGFACE_KEY
headers = {"Authorization": f"Bearer ${os.environ.get("HUGGINGFACE_KEY")}"}


def query(prompt, temperature=0.5):
    client = OpenAI(api_key=os.environ.get("OPENAI_KEY"))
    chat_completion = client.chat.completions.create(
            messages=[
                {"role": "system", "content": "Give python code only. No explaination."},
                {"role": "user", "content": prompt}
            ],
            model="gpt-3.5-turbo",
            max_tokens=1500,
            temperature=temperature,
    )
    return chat_completion


class RepeatedSample:

    def __init__(self):
        pass

    def inference(self, problem, num_solutions):
        # https://github.com/ScalingIntelligence/large_language_monkeys/blob/main/llmonk/generate/code_contests.py
        llama_prompt = ("Q: Write python code to solve the following coding "
                        "problem that obeys the constraints and passes the "
                        "example test cases. The output code needs to read "
                        "from and write to standard IO. Please wrap your code "
                        "answer using ```:")
        prompt = f"{llama_prompt}\n{problem}\nA: "
        solutions = []
        for _ in tqdm.trange(num_solutions):
            solution = query({"inputs": prompt})
            solutions.append(solution)
        return solutions


class SingleSample:

    def __init__(self):
        pass

    def inference(self, problem, num_solutions):
        # Slightly modified from
        # https://github.com/ScalingIntelligence/large_language_monkeys/blob/main/llmonk/generate/code_contests.py
        llama_prompt = (f"Repeat the following task {num_solutions} times: "
                        "Q: Write python code to solve the following coding "
                        "problem that obeys the constraints and passes the "
                        "example test cases. The output code needs to read "
                        "from and write to standard IO. Please wrap your code "
                        "answer using ```:")
        prompt = f"{llama_prompt}\n{problem}\nA: "
        solutions = []
        for _ in tqdm.trange(num_solutions):
            solution = query({"inputs": prompt})
            solutions.append(solution)
        return solutions


if __name__ == '__main__':
    model = RepeatedSample()
    two_sum = """
Given an array of integers nums and an integer target, return indices of the two numbers such that they add up to target.
You may assume that each input would have exactly one solution, and you may not use the same element twice.
You can return the answer in any order.
"""
    print(model.inference(two_sum, 1))
