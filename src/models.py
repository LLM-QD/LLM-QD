import os

from openai import OpenAI
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
import torch
import numpy as np


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


class NomicEmbedText:

    def __init__(self):
        self.model = SentenceTransformer("nomic-ai/nomic-embed-text-v1",
                                         trust_remote_code=True)

    def inference(self, sentence):
        # TODO Split the tokens properly
        tokens = [f"classification: {t}" for t in sentence.split(" ")]
        embeddings = self.model.encode(tokens, convert_to_tensor=True)
        return embeddings

class EmbeddingMatrix: 
    def __init__(self):
        self.tokenizer = AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3-8B-Instruct", use_auth_token=os.environ.get("HF_TOKEN"))
        self.model = AutoModelForCausalLM.from_pretrained("meta-llama/Meta-Llama-3-8B-Instruct", 
                                                          quantization_config= BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_use_double_quant=True, bnb_4bit_quant_type="nf4"),
                                                          use_auth_token=os.environ.get("HF_TOKEN"),
                                                          device_map="auto").to("cuda")
        self.model_embeddings = self.model.model.embed_tokens.weight.to('cuda')
        
    def tokenize(self, sentence):
        tokens = self.tokenizer(sentence, return_tensors="pt").to('cuda')
        embeddings = self.model.model.embed_tokens(tokens.input_ids).squeeze(0)
        return embeddings
    
    def decode(self,embeddings):
        new_tokens = []
        for token_embedding in embeddings:
            similarities = torch.matmul(token_embedding, self.model_embeddings.T)
            nearest_token_id = similarities.argmax().item()
            new_tokens.append(nearest_token_id)
        sentence = self.tokenizer.decode(new_tokens)
        return sentence

    
if __name__ == "__main__":
    two_sum = """Given an array of integers nums and an integer target, return indices of the two numbers such that they add up to target.
You may assume that each input would have exactly one solution, and you may not use the same element twice.
You can return the answer in any order.
"""

    # model = LLMSampler()

    model = NomicEmbedText()
    output = model.inference(two_sum)
    print(output)
