import json
import pandas as pd

def load_cot_dataset(json_path: str) -> pd.DataFrame:
   with open(json_path,"r",encoding="utf-8") as f:
       data = json.load(f)
       
   records = list(data.values())
   df = pd.DataFrame(records) 
   
   print(f"[DataLoader] Loaded {len(df)} samples, {df['task'].nunique()} tasks.")
   
   return df   