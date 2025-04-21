import subprocess

models = ["vicuna", "mistral", "llama"]
prompts = ["title_zero", "title_one", "wo_title_zero", "wo_title_one"]
num_versions = 3  # 可自定义生成几条描述

for model in models:
    for prompt in prompts:
        print(f"\n🚀 Running: Model={model} | Prompt={prompt}")
        subprocess.run([
            "python", "generate_prompts.py",
            "--prompt", prompt,
            "--model", model,
            "--num_versions", str(num_versions)
        ])
