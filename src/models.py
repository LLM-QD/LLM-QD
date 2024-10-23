import os

from openai import OpenAI


def query(prompt, temperature=0.5):
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


class RepeatedSample:
    def inference(self, prompt, num_solutions):
        # https://github.com/ScalingIntelligence/large_language_monkeys/blob/main/llmonk/generate/code_contests.py
        solutions = []
        for _ in range(num_solutions):
            solution = query(prompt)
            solutions.append(solution)
        return solutions


class SingleSample:
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
        for _ in range(num_solutions):
            solution = query(prompt)
            solutions.append(solution)
        return solutions


if __name__ == '__main__':
    model = RepeatedSample()
    two_sum = """
Given an array of integers nums and an integer target, return indices of the two numbers such that they add up to target.
You may assume that each input would have exactly one solution, and you may not use the same element twice.
You can return the answer in any order.
"""
    output = model.inference(two_sum, 1)
    print(output)
