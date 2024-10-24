import os

from openai import OpenAI


def query(prompt, temperature):
    # print(os.environ.get("OPENAI_KEY"))
    client = OpenAI(api_key=os.environ.get("OPENAI_KEY"),
                    base_url="https://api.deepinfra.com/v1/openai")
    response = client.chat.completions.create(
        model="meta-llama/Meta-Llama-3-8B-Instruct",
        messages=[{
            "role":
            "system",
            "content":
            "Give python code wrapped in '```python' code box. No explaination."
        }, {
            "role": "user",
            "content": prompt
        }],
        max_tokens=4096,
        temperature=temperature,
    )
    code_output = response.choices[0].message.content.strip()
    code_output = code_output.split("```python")[1].split("```")[0].strip()
    return code_output


class LLMSampler:

    def __init__(self, pre_prompt="", temperature=0.5):
        self.pre_prompt = pre_prompt
        self.temperature = temperature

    def inference(self, prompt, num_samples):
        prompt = f"{self.pre_prompt}\n{prompt}\nA: "
        solutions = []
        for _ in range(num_samples):
            solution = query(prompt, self.temperature)
            solutions.append(solution)
        return solutions


if __name__ == "__main__":
    model = LLMSampler()
    two_sum = """
Given an array of integers nums and an integer target, return indices of the two numbers such that they add up to target.
You may assume that each input would have exactly one solution, and you may not use the same element twice.
You can return the answer in any order.
"""
    output = model.inference(two_sum, 1)
    print(output)
