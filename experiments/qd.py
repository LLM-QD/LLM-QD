import subprocess

import numpy as np
import pandas as pd
import tqdm
from logdir import LogDir
from ribs.archives import GridArchive
from ribs.emitters import EvolutionStrategyEmitter
from ribs.schedulers import Scheduler

from src.dataloader import CodeContestsDataset
from src.metrics import get_diversities, pairwise_diversity
from src.models import LLMSampler, NomicEmbedText, EmbeddingMatrix

EmbedMatrix = EmbeddingMatrix()

def generate_solutions(model, description, k=5):
    solutions = model.inference(description, k)
    return solutions


def write_solutions_to_files(logdir, solutions, name):
    name = name.replace(" ", "_").replace(".", "")
    paths = []
    for i, solution in enumerate(solutions):
        path = logdir.file(f"{name}_{i}.py", touch=True)
        with open(path, "w", encoding="utf-8") as file:
            file.write(solution)
        paths.append(path)
    return paths


def evaluate_accuracy(solution_files, test_cases):
    """Evalutes the accuracy of a list of solutions for a problem.

    Args:
        solution_files (list of str): List of strings that represents paths to
            solution files to evaluate on the test cases.
        test_cases (dict):
            {"input": [input_1, input_2, ...], "output": [output_1, output_2, ...]}
    """
    metrics = {
        "solution_file": [],
        "test_input": [],
        "test_output": [],
        "model_output": [],
        "error": [],
        "passed": [],
    }

    for solution_file in solution_files:
        for test_input, expected_output in zip(test_cases["input"],
                                               test_cases["output"]):
            test_input = test_input.strip()
            expected_output = expected_output.strip()
            output = ""
            error = ""
            try:
                time_limit = 5
                result = subprocess.run(["python3", solution_file],
                                        input=test_input.encode(),
                                        capture_output=True,
                                        timeout=time_limit)
                output = result.stdout.decode().strip()
                error = result.stderr.decode().strip()
                # print("O:", output)
                # print("Expected:", expected_output)
                # print(output == expected_output)
                # print("Error:", error)
            except subprocess.TimeoutExpired:
                error = "TIMEOUT"

            # Log metrics.
            metrics["solution_file"].append(solution_file)
            metrics["test_input"].append(test_input)
            metrics["test_output"].append(expected_output)
            metrics["model_output"].append(output)
            metrics["error"].append(error)
            metrics["passed"].append(output == expected_output)
    return metrics

# Why does the scheduler need the initial code? 
def create_scheduler(init_prompt, init_code):
    ranges = [(0, 1)]
# Why is the solution dim 100? Should it not match the embedding dim?
    archive = GridArchive(solution_dim=100,
                          dims=(100, 100),
                          ranges=ranges,
                          learning_rate=0.01,
                          threshold_min=0.0)

    emb = EmbedMatrix.tokenize(init_prompt)

    # Do we need to flatten the embeddings for x0?  
    emitters = [
        EvolutionStrategyEmitter(
            archive,
            x0=emb,
            sigma0=0.5,
            ranker="imp",
            selection_rule="mu",
            restart_rule="basic",
            batch_size=36,
        ) for _ in range(15)
    ]

    scheduler = Scheduler(archive, emitters)
    return scheduler


def run_experiment(model_name, num_solutions, solution_files=None):
    # Load dataset.
    dataset = CodeContestsDataset()

    logdir = LogDir("[MODEL_NAME]")
    additional_info = [
        "Model: LLMQD",
        f"num_solutions: {num_solutions}",
    ]
    logdir.readme(date=True,
                  git_commit=True,
                  git_path="../",
                  info=additional_info)

    metrics = {
        "name": [],
        "solution_file": [],
        "test_input": [],
        "test_output": [],
        "model_output": [],
        "error": [],
        "passed": [],
    }

    embedding_model = NomicEmbedText()

    for name, desc, tests in dataset[:1]:
        print(name)

        # LLM model.
        pre_prompt = ("Q: Write python code to solve the following coding "
                      "problem that obeys the constraints and passes the "
                      "example test cases. The output code needs to read from "
                      "and write to standard IO. Please wrap your code answer "
                      "using ```:")
        temperature = 0.5
        llm = LLMSampler(pre_prompt, temperature=temperature)

        init_prompt = desc
        init_code = llm.inference(init_prompt, num_samples=1)

        # Create Scheduler.
        scheduler = create_scheduler(init_prompt=init_prompt,
                                     init_code=init_code)

        # QD loop.
        for i in tqdm.tqdm(range(100)):
            prompt_embeddings = scheduler.ask()
            codes = []
            for emb in prompt_embeddings:
                prompt = EmbedMatrix.decode(emb)
                # TODO: Remap tokens to vocab
                codes.append(llm.inference(prompt, num_samples=1))

            # Writes solutions to files.
            solution_files = write_solutions_to_files(logdir, codes, name)

            # Evaluate objective (accuracy of code).
            acc_metrics = evaluate_accuracy(solution_files,
                                            tests["private_tests"])

            # Evlauate features.
            features = np.empty((len(codes), 2))
            for j, code in enumerate(codes):
                structural_div, semantic_div = get_diversities(init_code, code)
                features[j] = [structural_div, semantic_div]

            scheduler.tell(acc_metrics, features)

        # Generate solutions.
        # TODO: Sample solutions from archive
        elites = scheduler.archive.sample_elites(num_solutions)
        solutions = elites["Solutions"]
        # Writes solutions to files.
        solution_files = write_solutions_to_files(logdir, solutions, name)

        # Evaluate accuracy of solution code.
        acc_metrics = evaluate_accuracy(solution_files, tests["private_tests"])

        # Log the metrics; one row per solution file per test.
        num_tests = len(tests["private_tests"]["input"])
        metrics["name"] += [name] * len(solution_files) * num_tests
        for key, value in acc_metrics.items():
            metrics[key] += value

        # Evaluate diversity.

    df = pd.DataFrame(metrics)
    df.to_csv(logdir.pfile("test_summary.csv", touch=True))
    df.to_pickle(logdir.pfile("test_summary.pkl", touch=True))


if __name__ == "__main__":
    num_solutions = 3
    run_experiment("single_sampler", num_solutions)
