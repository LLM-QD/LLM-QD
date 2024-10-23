# LLM-QD

## Install Dependencies

```bash
git clone git@github.com:LLM-QD/LLM-QD.git
cd LLM-QD

conda create -n llmqd
conda activate llmqd
pip install -e .
```

## Run Experiments

Run baseline experiments with **single sampler** and **repeated sampler**.
- Single sampler samples `meta-llama/Meta-Llama-3-8B-Instruct` with a single
prompt to generate $k$ solutions.
- Repeated sampler samples `meta-llama/Meta-Llama-3-8B-Instruct` $k$ times with
  a prompt prompt that generate *one* solution.

```bash
# Remember to activate conda environment.
cd experiments
python3 baselines.py
```

The experiment result will be stored in `experiments/logs/`.
