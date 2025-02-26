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
    "QA_queryCoT_to_answer": (
        "Task Description:\n1. Read the given question and related chain of thought to gather relevant information.\n2. The content of the chain of thought is the thinking process that may be used to answer the question.\n3. If the chain of thought don't work, please answer the question based on your own knowledge.\n4. Please answer the question and only output the answer.\n Question:{question}\nChain of Thought:{CoT}"
    ),
    "QA_queryCoT_to_answer_forrouge": (
        "Task Description:\n1. Read the given question and related chain of thought to gather relevant information.\n2. The content of the chain of thought is the thinking process that may be used to answer the question.\n3. If the chain of thought don't work, please answer the question based on your own knowledge.\n4. Give a short answer to a given question.\n Question:{question}\nChain of Thought:{CoT}"
    ),
    "QA_queryCoT_to_answer_forasqa": (
        "Task Description:\n1. Read the given question and related chain of thought to gather relevant information.\n2. The content of the chain of thought is the thinking process that may be used to answer the question.\n3. If the chain of thought don't work, please answer the question based on your own knowledge.\n4. Answer the following question. The question may be ambiguous and have multiple correct answers, and in that case, you have to provide a long-form answer including all correct answers.\n Question:{question}\nChain of Thought:{CoT}"
    ),
    "vanilla_RAG":(
        "Background:{Passages}\nQuestion:{Question}Answer:"
    ),
}
