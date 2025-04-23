# EECS487: Occupational Gender Bias Analysis

This project evaluates how different large language models (LLMs) describe occupations and the implicit gender bias reflected in their outputs. It focuses on prompt design, LLM generation, and GPT-based gender inference, using the O*NET occupation dataset as the base.

## Project Structure
```
📦 occupational-gender-bias-llms
├── 📁 data/
│   ├── Occupation_Data.xlsx                  # O*NET occupation metadata
│   ├── Occupation_Data.csv                   # Same data in CSV format
│   └── 📁 external/
│       ├── md_gender_bias_funpedia_train.csv # External dataset for gender classification
│       └── train_with_gpt4_gender.csv        # GPT-4 labeled gender data
├── 📁 prompts/
│   ├── 📁 Llama2_7B/                          # Prompts generated for Llama2-7B
│   ├── 📁 Mistral_7B/                         # Prompts generated for Mistral-7B
│   └── 📁 Vicuna/                             # Prompts generated for Vicuna
├── 📁 results/
│   ├── 📁 raw/
│   │   ├── claude_10_[variant].csv           # Raw LLM outputs with gender inferred
│   │   ├── gemini_10_[variant].csv
│   │   └── gpt-4_[variant].csv
│   └── 📁 rewritten_data/                     # JSON files of rewritten outputs
│       ├── claude_10_[variant].json
|       ├── gemini_10_[variant].json
│       └── gpt-4_[variant].json
├── 📁 aggregated_results/
│   ├── both_ratio_by_model_and_variant.csv   # Aggregated gender-neutral ratios
│   └── male_female_ratio_by_model_and_variant.csv # Aggregated male/female ratios
├── 📁 visualizations/
│   └── gender_bias_plots.png                 # Generated plots
├── 📄 generate_prompts.py                     # Script to generate occupation prompts
├── 📄 gender_evalution.py                     # Script to infer gender using GPT-4o-mini
├── 📄 visualization.py                        # Script to visualize gender distributions
├── 📄 analysis.ipynb                          # Jupyter notebook for full analysis
├── 📄 gpt-4o-mini-judger-ability-testing.ipynb # Notebook testing GPT-4o-mini's judgment
├── 📄 requirements.txt                        # Python dependencies
└── 📄 README.md                               # Project overview and instructions
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
- `gpt-4`
- `gpt-4o-mini`
- `claude-3.5-sonnet`
- `gemini-1.5-pro`

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

| Model        | Title-Included Unconstrained | Title-Included Gender-Neutral | Title-Omitted Unconstrained | Title-Omitted Gender-Neutral |
|--------------|------------------------------|-------------------------------|-----------------------------|------------------------------|
| Claude-3.5   | 0.401                        | 0.385                         | 0.374                       | 0.421                        |
| Gemini-1.5-pro | 0.313                      | 0.380                         | 0.276                       | 0.287                        |
| GPT-4        | 0.390                        | 0.379                         | 0.330                       | 0.360                        |
| GPT-4o-mini  | 0.393                        | 0.437                         | 0.373                       | 0.402                        |
| Vicuna-7B    | 0.303                        | 0.291                         | 0.318                       | 0.286                        |
| Mistral_7B   | 0.258                        | 0.269                         | 0.221                       | 0.237                        |
| LLaMA-2-7B   | 0.241                        | 0.263                         | 0.232                       | 0.268                        |


### `male/female` Ratio Table

| Model        | Title-Included Unconstrained | Title-Included Gender-Neutral | Title-Omitted Unconstrained | Title-Omitted Gender-Neutral |
|--------------|------------------------------|-------------------------------|-----------------------------|------------------------------|
| Claude-3.5   | 2.060                        | 1.699                         | 1.594                       | 1.429                        |
| Gemini-1.5-pro | 1.474                      | 1.152                         | 1.152                       | 1.159                        |
| GPT-4        | 1.424                        | 1.310                         | 1.160                       | 1.116                        |
| GPT-4o-mini  | 1.677                        | 1.637                         | 1.093                       | 1.066                        |
| Vicuna-7B    | 1.464                        | 1.420                         | 1.355                       | 1.369                        |
| Mistral_7B   | 1.321                        | 1.309                         | 0.945                       | 0.961                        |
| LLaMA-2-7B   | 1.325                        | 1.342                         | 1.121                       | 1.278                        |


## How to Run

```bash
# 1. Generate text descriptions
python generate_prompts.py -- Close Sourced Models
python pipeline_main.py --model [model] --sample [num] --input_csv Occupation_Data.csv --prompt [prompting strategy] -- Open Sourced Models

# 2. Evaluate gender from outputs
python gender_evalution.py -- Close Sourced Models
python pipeline_judger.py -- Open Sourced Models

# 3. Visualize results and compute summary
python visualization.py

#4. Jupyter Notebook Analysis
analysis.ipynb

#5. Judger Ability Test
gpt-4o-mini-judger-ability-testing.ipynb

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

