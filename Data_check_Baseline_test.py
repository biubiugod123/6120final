
import re
import json
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from tqdm.auto import tqdm



with open("archive (1)/CoT_collection.json", "r", encoding="utf-8") as f:
    data = json.load(f)

records = list(data.values())

df = pd.DataFrame(records)

print(df.shape)
print(df.head())
print(df.columns)

# num of samples
num_samples = len(df)
print("Number of samples:", num_samples)


num_tasks = df["task"].nunique()
print("Number of tasks:", num_tasks)

task_counts = df["task"].value_counts()
print(task_counts.head(20))


task_dist = df["task"].value_counts(normalize=True)
print(task_dist.head(20))

df["rationale_len_tokens"] = df["rationale"].astype(str).str.split().str.len()
print(df["rationale_len_tokens"].describe())

df["source_len_tokens"] = df["source"].astype(str).str.split().str.len()
print(df["source_len_tokens"].describe())

task_cot_mean = df.groupby("task")["rationale_len_tokens"].mean().sort_values(ascending=False)
print(task_cot_mean.head(10))





tasks_for_eval = ["math_dataset", "quoref", "drop"]  

dfs = {t: df[df["task"] == t].copy() for t in tasks_for_eval}
for t, dft in dfs.items():
    print(t, len(dft))



def normalize_text(s):
    s = str(s).strip().lower()
    s = re.sub(r'\.$', '', s)      
    s = re.sub(r'\s+', ' ', s)     
    return s


#!pip install transformers accelerate


model_name = "google/flan-t5-base"
device = "cuda" if torch.cuda.is_available() else "cpu"

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name).to(device)
model.eval()


def eval_baseline_flan(df_eval, max_samples=500, batch_size=8,
                       max_input_len=512, max_new_tokens=64):
    
    n = min(max_samples, len(df_eval))
    df_sample = df_eval.sample(n, random_state=0).reset_index(drop=True)
    
    preds = []
    refs = []
    
    for i in tqdm(range(0, n, batch_size)):
        batch = df_sample.iloc[i:i+batch_size]
        inputs = batch["source"].tolist()     # input: source
        targets = batch["target"].tolist()    # true answer: target

        enc = tokenizer(
            inputs,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_input_len
        ).to(device)

        with torch.no_grad():
            outputs = model.generate(
                **enc,
                max_new_tokens=max_new_tokens
            )

        decoded = tokenizer.batch_decode(outputs, skip_special_tokens=True)

        preds.extend(decoded)
        refs.extend(targets)

    #  Exact Match accuracy
    correct = 0
    for p, r in zip(preds, refs):
        if normalize_text(p) == normalize_text(r):
            correct += 1

    acc = correct / len(refs)
    return acc, preds, refs




tasks_for_eval = ["math_dataset", "quoref", "drop"]

results = []

for task_name in tasks_for_eval:
    df_task = df[df["task"] == task_name]
    if len(df_task) == 0:
        print(f"Task {task_name} has no data, skip.")
        continue
        
    acc, preds, refs = eval_baseline_flan(df_task, max_samples=500)
    print(f"Baseline (Flan-T5-base) on {task_name}: EM = {acc:.3f} over {len(refs)} examples")
    
    results.append({
        "task": task_name,
        "accuracy": acc,
        "num_samples": len(refs),
    })



import matplotlib.pyplot as plt


import pandas as pd
plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams["figure.facecolor"] = "white"
res_df = pd.DataFrame(results)

res_df = res_df.sort_values("accuracy", ascending=False)

plt.figure(figsize=(6, 5))
colors = ["red", "navy", "green"]
bar_colors = [colors[i % len(colors)] for i in range(3)]

plt.bar(res_df["task"], res_df["accuracy"],width=0.4, color=bar_colors)
plt.ylim(0, 1.0)  
plt.xlabel("Task")
plt.ylabel("Exact Match Accuracy")
plt.title("Baseline (Flan-T5-base) EM on CoT Collection Tasks")


for i, row in res_df.iterrows():
    plt.text(
        x=row["task"],
        y=row["accuracy"] + 0.01,
        s=f"{row['accuracy']:.2f}",
        ha="center",
        va="bottom",
        fontsize=10,
        rotation=0
    )

plt.tight_layout()
plt.show()

