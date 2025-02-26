import pandas as pd
import json


jsonl_file_path = "nq_queryCOT_to_answer.jsonl"

n = 0
x = 0
# Read JSON Lines file and create a DataFrame
json_lines = []
with open(jsonl_file_path, 'r', encoding='utf-8') as jsonl_file:
    for line in jsonl_file:
        n+=1
        data = json.loads(line)
        ground_truth = data['ground_truth']
        answers = str(data['model_answer'])
        if any(answer.lower() in answers.lower() for answer in ground_truth):
            x+=1

print(x/n)