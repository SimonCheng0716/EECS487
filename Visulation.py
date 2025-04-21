import os
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

def progress(data1, data2):
    merged_data = pd.merge(data1, data2, left_on='title', right_on='Title', how='left')
    data1['code'] = merged_data['O*NET-SOC Code']
    data1['code'] = data1['code'].astype(str) + '-' + (data1['version'] - 1).astype(str)
    return data1[['code', 'gender']].rename(columns={'code': 'ID', 'gender': 'Gender'})

def plot(data, model, output_folder):
    df = data.copy()
    df['Group'] = df['ID'].str.extract(r'^(\d+)-')
    group_mapping = {
        '11': 'Management', '13': 'Business & Financial Ops', '15': 'Computer & Math',
        '17': 'Architecture & Engineering', '19': 'Life, Physical & Social Science',
        '21': 'Community & Social Services', '23': 'Legal',
        '25': 'Education, Training & Library', '27': 'Arts, Design, Entertainment, Sports, Media',
        '29': 'Healthcare Practitioners & Technical', '31': 'Healthcare Support',
        '33': 'Protective Service', '35': 'Food Prep & Serving',
        '37': 'Building & Grounds Cleaning', '39': 'Personal Care & Service',
        '41': 'Sales', '43': 'Office & Admin Support', '45': 'Farming, Fishing & Forestry',
        '47': 'Construction & Extraction', '49': 'Installation, Maintenance & Repair',
        '51': 'Production', '53': 'Transportation & Material Moving', '55': 'Military Specific'
    }
    df['Occupation'] = df['Group'].map(group_mapping)
    counts = df.groupby(['Occupation', 'Gender']).size().unstack(fill_value=0)

    os.makedirs(output_folder, exist_ok=True)
    ax = counts.plot(kind='bar', stacked=False, figsize=(12, 6))
    ax.set_title(f'Gender Distribution by Occupation Group ({model})')
    ax.set_xlabel('Occupation')
    ax.set_ylabel('Count')
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')
    ax.legend(title='Gender')
    ax.grid(axis='y', linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.savefig(f"{output_folder}/{model}.png", dpi=300)
    plt.close()

# === Main Execution ===
csv_folder = Path("/content/drive/MyDrive/EECS487/pro")
output_folder = csv_folder / "results"
data_path = csv_folder / "data/Occupation Data.xlsx"
input_files = list(csv_folder.glob("*_sample_10_des_gender_eval.csv"))

# Prepare structure
male_female_ratios = {}
both_ratios = {}

# Load occupation metadata once
occupation_data = pd.read_excel(data_path)

# Loop through all CSVs
for file in input_files:
    filename = file.name
    model_type = filename.replace("_sample_10_des_gender_eval.csv", "")
    print(f"Processing {model_type}...")

    df = pd.read_csv(file)
    processed = progress(df, occupation_data)
    plot(processed, model_type, output_folder)

    # Compute ratios
    total = len(processed)
    male = (processed['Gender'] == 'male').sum()
    female = (processed['Gender'] == 'female').sum()
    both = (processed['Gender'] == 'both').sum()

    # Extract model + prompt_type
    model_parts = model_type.split("_")
    model = "_".join(model_parts[:2])  # e.g., vicuna_7B
    prompt_type = "_".join(model_parts[2:])  # e.g., title_zero

    # Fill ratio dicts
    both_ratios.setdefault(model, {})[prompt_type] = both / total if total else 0
    mf_ratio = male / female if female else float("nan")
    male_female_ratios.setdefault(model, {})[prompt_type] = mf_ratio

# === Generate Tables ===
desired_order = ["title_zero", "title_one", "wo_title_zero", "wo_title_one"]
df_both = pd.DataFrame(both_ratios).T.reindex(columns=desired_order)
df_mf = pd.DataFrame(male_female_ratios).T.reindex(columns=desired_order)

# === Save and Print ===
df_both.to_csv(output_folder / "both_ratio_by_model_and_variant.csv", float_format="%.4f")
df_mf.to_csv(output_folder / "male_female_ratio_by_model_and_variant.csv", float_format="%.4f")

print("Both Ratio Table:")
print(df_both)

print("Male/Female Ratio Table:")
print(df_mf)
