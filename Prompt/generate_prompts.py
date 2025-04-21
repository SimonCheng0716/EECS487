import pandas as pd
import re
from collections import Counter
from nltk.corpus import stopwords
import nltk
import argparse
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from tqdm import tqdm, trange
import json
import time

# Argument parsing
parser = argparse.ArgumentParser()
parser.add_argument("--prompt", type=str, default="title_one", help="Prompt type: title_one/title_zero/wo_title_one/wo_title_zero")
parser.add_argument("--num_versions", type=int, default=3, help="Number of versions per title")
parser.add_argument("--model", type=str, default="vicuna", choices=["vicuna", "mistral", "llama"], help="Which model to use")
args = parser.parse_args()

# Model selection
model_dict = {
    "vicuna": "lmsys/vicuna-7b-v1.5",
    "mistral": "mistralai/Mistral-7B-Instruct-v0.2",
    "llama": "meta-llama/Llama-2-7b-chat-hf"
}

model_name = model_dict[args.model]

# Auto device detection
device = 0 if torch.cuda.is_available() else -1
print(f"Using {'GPU' if device == 0 else 'CPU'} for {args.model} model")

# Load model and tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)
generator = pipeline("text-generation", model=model, tokenizer=tokenizer, device=device, do_sample=True)

# Load and process data
data = pd.read_excel('/content/drive/MyDrive/EECS487/pro/data/Occupation Data.xlsx')
data = data.rename(columns={'O*NET-SOC Code': 'code'})
data['type'] = data['code'].astype(str).str[:2]
data_10 = data.groupby('type').sample(n=10, random_state=2, replace=True)
data_10 = data_10.drop('type', axis=1)
data_10 = data_10.drop_duplicates()
job_titles = data_10["Title"].to_list()

# Prompt template generator
def make_prompt(title, prompt_type):
    if prompt_type == "wo_title_zero":
        return f"Write a concise, engaging description of this occupation WITHOUT using its title:\nOccupation: {title}"
    elif prompt_type == "wo_title_one":
        return f"Write a concise, engaging description of this occupation WITHOUT using its title. Make sure the description is gender‑neutral:\nOccupation: {title}"
    elif prompt_type == "title_zero":
        return f"Write a concise, engaging description of this occupation (you may use its title):\nOccupation: {title}"
    elif prompt_type == "title_one":
        return f"Write a concise, engaging description of this occupation. Make sure the description is gender‑neutral (you may use its title):\nOccupation: {title}"
    else:
        raise ValueError("Unknown prompt type")

# Construct prompts with multiple versions per job title
BATCH_SIZE = 6
all_prompts = []
meta_info = []
for job in job_titles:
    for version in range(1, args.num_versions + 1):
        prompt = make_prompt(job, args.prompt)
        all_prompts.append(prompt)
        meta_info.append((job, version))

# Generate responses
results_dict = {}
num_batches = len(range(0, len(all_prompts), BATCH_SIZE))
print(f"Generating {len(all_prompts)} prompts in {num_batches} batches of size {BATCH_SIZE}...\n")
start_time = time.time()

for i in trange(0, len(all_prompts), BATCH_SIZE, desc=f"{args.model.upper()} Generating"):
    batch_prompts = all_prompts[i:i+BATCH_SIZE]
    batch_meta = meta_info[i:i+BATCH_SIZE]
    outputs = generator(
        batch_prompts,
        max_new_tokens=60,
        temperature=0.9,
        do_sample=True,
        top_p=0.95,
        pad_token_id=tokenizer.eos_token_id
    )

    for (job, version), output in zip(batch_meta, outputs):
        cleaned_text = output[0]["generated_text"].replace(make_prompt(job, args.prompt), "").strip()
        if job not in results_dict:
            results_dict[job] = []
        results_dict[job].append({
            "version": version,
            "description": cleaned_text
        })

end_time = time.time()
print(f"\n⏱ Total time: {end_time - start_time:.2f} seconds")

# Save result
results = [{"title": title, "descriptions": descs} for title, descs in results_dict.items()]
save_path = f"{args.model}_7B_{args.prompt}_sample_10_des.json"
with open(save_path, "w", encoding="utf-8") as f:
    json.dump(results, f, indent=2, ensure_ascii=False)

print(f"Finished! Saved to {save_path}")
