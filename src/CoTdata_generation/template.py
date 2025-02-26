PROMPT_DICT = {
    "QA_querypassage_to_CoT": (
        "Passage:{passage}\nBased on these passages, answer the question below.\nQuestion:{question}\nLet's think step by step."
    ),
    "Mutichoice_querypassage_to_CoT": (
        "Passage:{passage}\nBased on these passages, please answer the multiple choice question below.\nQuestion:{question}\nLet's think step by step."
    ),
    "QA_queryCoT_to_answer": (
        "Task Description:\n1. Read the given question and related chain of thought to gather relevant information.\n2. The content of the chain of thought is the thinking process that may be used to answer the question.\n3. If the chain of thought don't work, please answer the question based on your own knowledge.\n4. Give a short answer to a given question.\n Question:{question}\nChain of Thought:{CoT}"
    ),
    "Mutichoice_queryCoT_to_answer": (
        "Task Description:\n1. Read the given question and related chain of thought to gather relevant information.\n2. The content of the chain of thought is the thinking process that may be used to answer the question.\n3. If the chain of thought don't work, please answer the question based on your own knowledge.\n4.Output only the choice between ***** and ***** at first,then give a short answer to the mutiple choice question.\n Question:{question}\nChain of Thought:{CoT}"
    ),
    
}