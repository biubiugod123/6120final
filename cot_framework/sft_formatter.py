from cot_framework.prompt_template import build_cot_prompt

def format_sft_pair(source: str, rationale: str, answer: str):
    """
    将一条 (source, rationale, answer) 转成 SFT 的 (input, target) 文本对。
    """
    # 输入给模型的 prompt（包含问题）
    sft_input = build_cot_prompt(source)
    # 模型要学习输出的内容：推理过程 + 最终答案
    sft_target = f"{rationale}\n\nFinal Answer: {answer}"
    return sft_input, sft_target


def build_sft_dataset(df):
    """
    把整个 DataFrame 转成一个 list[{"input": ..., "target": ...}]
    方便后续保存为 jsonl 做 SFT。
    """
    formatted = []
    
    for _, row in df.iterrows():
        inp, tgt = format_sft_pair(
            row["source"],
            row["rationale"],
            row["target"]
        )
        formatted.append({"input": inp, "target": tgt})
        
    print(f"[SFT Formatter] Formatted {len(formatted)} samples.")
    return formatted
