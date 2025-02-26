import json
import random


input_file = 'llama3ft_dpodata_notnone.jsonl'

train_file = 'llama3ft_dpodata_train.jsonl'
valid_file = 'llama3ft_dpodata_validation.jsonl'

with open(input_file, 'r', encoding='utf-8') as file:
    lines = file.readlines()


random.shuffle(lines)
total_lines = len(lines)
train_end = int(total_lines * 0.9)
valid_end = train_end + int(total_lines * 0.1)


train_data = lines[:train_end]
valid_data = lines[train_end:]


with open(train_file, 'w', encoding='utf-8') as out_file:
    for line in train_data:
        out_file.write(line)


with open(valid_file, 'w', encoding='utf-8') as out_file:
    for line in valid_data:
        out_file.write(line)
