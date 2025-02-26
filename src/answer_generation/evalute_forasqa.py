import json
import string
import re
import numpy as np
import copy
def normalize_answer(s):
    def remove_articles(text):
        return re.sub(r"\b(a|an|the)\b", " ", text)

    def white_space_fix(text):
        return " ".join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))

def compute_str_em(data):
    """Compute STR-EM metric (only for ASQA)
    Args:
        data: requires field `qa_pairs/short_answers` and `model_answer`
    Returns:
        STR-EM and STR-EM-HIT ()
    """

    if 'qa_pairs' not in data[0] or data[0]['qa_pairs'] is None:
        return 0, 0

    acc = []
    hit = []

    for item in data:
        loc_acc = []
        for qa_pair in item['qa_pairs']:
            loc_acc.append(exact_presence(qa_pair['short_answers'], item["model_answer"]))
        acc.append(np.mean(loc_acc))
        hit.append( int(np.mean(loc_acc) == 1) )

    return 100 * np.mean(acc), 100 * np.mean(hit)


def exact_presence(short_answers, context):
    """Verify if any of the answers is present in the given context.
    Args:
        short_answers: list of short answers to look for in the context
        context: a paragraph to search for short answers
    Returns:
        true if any of the short answers is present in the context
    """

    n_short_answers = [normalize_answer(sa) for sa in short_answers]
    n_context = normalize_answer(context)

    for ans in n_short_answers:
        if ans in n_context:
            return True

    return False

def remove_citations(sent):
    return re.sub(r"\[\d+", "", re.sub(r" \[\d+", "", sent)).replace(" |", "").replace("]", "")


file_path = 'asqa_queryCOT_to_answer.jsonl'  
data = []
with open(file_path, 'r', encoding='utf-8') as f:
    for line in f:
        data.append(json.loads(line.strip()))
        
for i in range(len(data)):
        # data[i]['model_answer'] = data[i]['model_answer'].strip().split("\n")[0]
        data[i]['model_answer'] = data[i]['model_answer'].replace("<|im_end|>", "")

normalized_data = copy.deepcopy(data)
for i in range(len(normalized_data)):
    normalized_data[i]['model_answer'] = remove_citations(normalized_data[i]['model_answer'])
str_em, str_hit = compute_str_em(normalized_data)
print(f"strem: {str_em}")
print(f"strhit: {str_hit}")