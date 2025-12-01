from cot_framework.data_loader import load_cot_dataset
from cot_framework.sft_formatter import build_sft_dataset


if __name__ == "__main__":
    import os
    print(">>> CWD =", os.getcwd())
    print(">>> Writing to:", os.path.abspath("data/cot_sft_formatted.jsonl"))
   
    # 1. Load raw dataset
    df = load_cot_dataset("data/CoT_collection.json")
    
    # 2. Filter some tasks
    tasks = ["math_dataset", "drop", "quoref"]
    df = df[df["task"].isin(tasks)]
    print(f"[Pipeline] Filtered dataset size: {len(df)}")
    
    # 3. Format for SFT
    formatted = build_sft_dataset(df)
    
    #4. Save formatted file
    import jsonlines
    with jsonlines.open("data/cot_sft_formatted.jsonl","w") as writer:
        writer.write_all(formatted)
        
    print("[Pipeline] Saved formatted dataset -> data/cot_sft_formatted.jsonl")     
    
    