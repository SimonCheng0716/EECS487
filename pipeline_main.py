import argparse
import csv
import json
import logging
import random
import os
import google.generativeai as genai

gpt_api_keys = "YOUR-OPENAI-API-KEY"
ANTHROPIC_API_KEY = "YOUR-ANTHROPIC-API-KEY"
genai.configure(api_key="YOUR-GEMINI-API-KEY")
gemini_model_name = "gemini-1.5-pro"
gemini_model = genai.GenerativeModel(gemini_model_name)

import openai
import anthropic

def query_llm(prompt, model):
    """
    Queries the specified language model with the given prompt.
    """
    global gpt_call_count

    if model.startswith("gpt"):
        openai.api_key = gpt_api_keys
        try:
            response = openai.ChatCompletion.create(
                model=model,
                messages=[
                    {"role": "system", "content": "You are a helpful agent that helps write the description for a given occupation"},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=60,
                temperature=0.4,
                n=1
            )
            return response.choices[0].message.content.strip()
        except openai.error.OpenAIError as e:
            logging.error(f"OpenAI API error: {e}")
            return None

    elif model.startswith("claude"):
        try:
            client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)
            response = client.messages.create(
                model="claude-3-5-sonnet-20241022",
                max_tokens=60,
                temperature=0.4,
                system="You are a helpful agent that helps write the description for a given occupation.",
                messages=[{"role": "user", "content": prompt}]
            )
            if isinstance(response.content, list):
                return "\n".join(block.text for block in response.content if block.type == 'text')
        except Exception as e:
            logging.error(f"Anthropic API error: {e}")
            return None

    elif model.startswith("gemini"):
        try:
            full_prompt = f"System: You are a helpful agent that helps write the description for a given occupation.\nUser: {prompt}"
            generation_config = genai.types.GenerationConfig(
                temperature=0.4,
                max_output_tokens=60
            )
            response = gemini_model.generate_content(full_prompt, generation_config=generation_config)
            return response.text.strip()
        except Exception as e:
            logging.error(f"Google Gemini API error: {e}")
            return None


    else:
        raise NotImplementedError(f"Model '{model}' is not implemented yet.")

def main():
    parser = argparse.ArgumentParser(
        description="Generate multiple occupation descriptions using a chosen LLM."
    )
    parser.add_argument(
        "--model", type=str, required=True,
        help="Which LLM model to use (e.g., gpt-3.5-turbo, llama-2-7b, etc.)"
    )
    parser.add_argument(
        "--sample", type=str, default="all",
        help="Number of occupations to sample per group, or 'all' for the entire dataset."
    )
    parser.add_argument(
        "--input_csv", type=str, required=True,
        help="Path to the input CSV file containing occupation data."
    )
    parser.add_argument(
        "--prompt", type=str, required=True,
        help="Choosing the prompting strategy."
    )
    args = parser.parse_args()

    occupations = []
    with open(args.input_csv, mode="r", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        for row in reader:
            occupations.append(row)

    grouped = {}
    for occ in occupations:
        gid = occ["O*NET-SOC Code"].split("-")[0]
        grouped.setdefault(gid, []).append(occ)

    if args.sample.lower() == "all":
        final_list = occupations
    else:
        cnt = int(args.sample)
        final_list = []
        for occs in grouped.values():
            final_list.extend(occs if len(occs) <= cnt else random.sample(occs, cnt))

    results = []
    for occ in final_list:
        title = occ["Title"]
        code = occ["O*NET-SOC Code"]
        descs = []
        for _ in range(3):
            if args.prompt == "wo_title_zero":
                prompt = (
                    f"Write a concise, engaging description of this occupation WITHOUT using its title:\n"
                    f"Occupation: {title}"
                )
            
            elif args.prompt == "wo_title_one":
                prompt = (
                    f"Write a concise, engaging description of this occupation WITHOUT using its title. "
                    f"Make sure the description is gender‑neutral:\n"
                    f"Occupation: {title}"
                )
            
            elif args.prompt == "title_zero":
                prompt = (
                    f"Write a concise, engaging description of this occupation (you may use its title):\n"
                    f"Occupation: {title}"
                )
            
            elif args.prompt == "title_one":
                prompt = (
                    f"Write a concise, engaging description of this occupation. "
                    f"Make sure the description is gender‑neutral (you may use its title):\n"
                    f"Occupation: {title}"
                )
            d = query_llm(prompt, args.model)
            print(d)
            descs.append(d or "No response from LLM")
        results.append({"Occupation": title, "Description": descs, "ID": code})

    output_dir = "rewritted_data"
    os.makedirs(output_dir, exist_ok=True)

    sample_label = args.sample if args.sample.lower() != "all" else "all"
    output_filename = f"{args.model}_{sample_label}_{args.prompt}.json"
    output_path = os.path.join(output_dir, output_filename)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print(f"Saved results to {output_path}")

if __name__ == "__main__":
    main()
