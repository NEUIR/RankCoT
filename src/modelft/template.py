IGNORE_INDEX = -100

user_tokens=[1786, 4194, 95388]  # <用户>

assistant_tokens=[1786, 10850, 95388] # <AI>

pythia_user_tokens=[29, 12335, 46136, 31]  # <用户>

pythia_assistant_tokens=[29, 18128, 31] # <AI>

RESPONSE_START_TOKEN_IDS = [128006, 78191, 128007]  # <|start_header_id|>assistant<|end_header_id|>

PROMPT_DICT = {
    "QA_querypassage_to_CoT": (
        "Passages:{passages}\nBased on these passages, answer the question below.\nQuestion:{question}\nLet's think step by step."
    ),
    "Mutichoice_querypassage_to_CoT": (
        "Passages:{passages}\nBased on these passages, please answer the multiple choice question below.\nQuestion:{question}\nLet's think step by step."
    ),
}
