import json

input_file = "data/cot_sft_formatted.jsonl"
output_file = "data/cot_sft_fixed.jsonl"

with open(input_file, "r") as f_in, open(output_file, "w") as f_out:
    for line in f_in:
        item = json.loads(line)
        new_item = {
            "instruction": "",
            "input": item.get("input", ""),
            "output": item.get("target", "")
        }
        f_out.write(json.dumps(new_item) + "\n")

print("转换完成 → 写入 data/cot_sft_fixed.jsonl")
