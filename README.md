# CS6120 Final Project: Chain-of-Thought Distillation with DPO & ORPO

This repository contains the implementation for a Natural Language Processing project focused on enhancing **Chain-of-Thought (CoT)** reasoning capabilities in smaller models (Flan-T5-Base).
## Download Required Files

**Crucial Step:** Before running any scripts, you must download the `data` and `results` folders from Google Cloud. These files are too large to host on GitHub.

| Resource | Description                                   | Download Link |
| :--- |:----------------------------------------------| :--- |
| **Data** | Contains all `.jsonl` training and test sets. | **[🔗 https://drive.google.com/file/d/1do2c6ktCBUC4oEnwadyXOMTTfsmCJUM7/view?usp=sharing]** |
| **Results** | Contains the trained PPO & DPO & ORPO model.  | **[🔗 https://drive.google.com/file/d/1pb9ygCce_3j1sSegDZyiY8mVm0Q7x1WH/view?usp=sharing]** |
## Project Structure

Below is the key file structure for this project. Please ensure your directories match this layout, particularly the `data/` and `results/` folders.
```text
6120final/
├── data/                       <-- [REQUIRED] Dataset Folder
│   ├── CoT_collection.json
│   ├── cot_rl_train.jsonl 
│   ├── cot_sft_fixed.jsonl
│   ├── cot_sft_formatted.jsonl
│   ├── cot_sft_train.jsonl
│   └── cot_test.jsonl 
│
├── results/                 
│   ├── rl_dpo/                 <-- DPO Model weights
│   ├── rl_orpo/                <-- ORPO Model weights
│   └── rl_ppo/                 <-- PPO Model weights
│
└── evaluate_rl.py              <-- Evaluation Script
```

## Requirements

- **transformers**
- **trl**
- **accelerate**
- **peft**
- **datasets**

## Run

To evaluate the RL results, run:

```bash
python evaluate_rl.py

