# EECS487: Occupational Gender Bias Analysis

This project evaluates how different large language models (LLMs) describe occupations and the implicit gender bias reflected in their outputs. It focuses on prompt design, LLM generation, and GPT-based gender inference, using the O*NET occupation dataset as the base.

## Project Structure

```
├── Data/
│   └── Occupation Data.xlsx              # O*NET occupation metadata
├── Prompt/
│   ├── [Model]/
├── Results/
│   ├── [Model]/
│   │   ├── CSV/                          # Gender evaluation CSVs
│   │   └── Visualization/                # Bar chart plots per occupation group
├── generate_prompts.py                   # Generate prompts using LLMs
├── gender_evalution.py                   # Use GPT-4o-mini to infer gender from LLM output
├── visualization.py                      # Generate plots and summarize ratios
├── README.md
```

## Prompting Strategies

Each job description is generated under four prompt types:

- `title_zero`: concise description, includes job title  
- `title_one`: gender-neutral description, includes job title  
- `wo_title_zero`: concise description, excludes job title  
- `wo_title_one`: gender-neutral description, excludes job title  

Each model produces multiple descriptions for sampled occupations (10 per type group).

## LLMs Used for Generation

- `Vicuna-7B`
- `Mistral-7B`
- `LLaMA2-7B-Chat`

## Gender Inference

Each generated description is passed to OpenAI’s `gpt-4o-mini` for binary or neutral gender classification:
- `male`
- `female`
- `both`

## Visualization Output

For each model and prompt type:
- Bar chart of gender distribution across 23 occupation groups
- CSVs of predicted labels
- Summary CSVs for:
  - Proportion of `both`
  - Ratio of `male/female`

Example Summary Tables:

### `both` Proportion Table

| model      | title_zero | title_one | wo_title_zero | wo_title_one |
|------------|------------|-----------|----------------|---------------|
| vicuna_7B  | 0.30       | 0.29      | 0.32           | 0.29          |
| mistral_7B | 0.26       | 0.27      | 0.28           | 0.28          |
| llama_7B   | 0.31       | 0.30      | 0.33           | 0.30          |

### `male/female` Ratio Table

| model      | title_zero | title_one | wo_title_zero | wo_title_one |
|------------|------------|-----------|----------------|---------------|
| vicuna_7B  | 1.10       | 1.12      | 0.95           | 1.00          |
| mistral_7B | 1.20       | 1.18      | 1.05           | 1.08          |
| llama_7B   | 1.15       | 1.14      | 1.01           | 1.03          |

## How to Run

```bash
# 1. Generate text descriptions
python generate_prompts.py

# 2. Evaluate gender from outputs
python gender_evalution.py

# 3. Visualize results and compute summary
python visualization.py
```

## Requirements

```bash
pip install -r requirements.txt
```

Include:
- pandas
- matplotlib
- openai
- transformers
- tqdm

## Author

Created by [Simon Cheng](https://github.com/SimonCheng0716), Xiaochun Wei for EECS 487 at the University of Michigan.

