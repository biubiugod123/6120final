def build_cot_prompt(source: str) -> str:
    prompt = (
        "You are a helpful reasoning assistant.\n"
        "Please reason step-by-step and provide the final answer.\n\n"
        f"Question:{source}\n\nAnswer:"
    )
    return prompt