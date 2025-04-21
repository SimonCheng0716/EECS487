import subprocess

models = ["vicuna", "mistral", "llama"]
prompts = ["title_zero", "title_one", "wo_title_zero", "wo_title_one"]
num_versions = 3

for model in models:
    for prompt in prompts:
        print(f"Running: Model={model} | Prompt={prompt}")
        subprocess.run([
            "python", "generate_prompt.py",
            "--prompt", prompt,
            "--model", model,
            "--num_versions", str(num_versions)
        ])
