import argparse
from rouge import Rouge
import json
import random
from tqdm import tqdm
random.seed(1)


def _rougel_score(prediction, ground_truth):
    rouge = Rouge()
    # no normalization
    try:
        scores = rouge.get_scores(prediction, ground_truth, avg=True)
    except ValueError:  # "Hypothesis is empty."
        return 0.0
    return scores["rouge-l"]["f"]

def custom_json_decoder(obj):
    if 'id' in obj:
        obj['id'] = str(obj['id'])
    return obj

# 输入文件路径
input_file = '/home/wumingyan/DDR_RCD_aligned/COTdata_generation/llama3_interf/result/queryCOT_to_answer.jsonl'
# 输出文件路径
output_file = '/home/wumingyan/DDR_RCD_aligned/modelft/data/llama3ft_losssum_dpodata.jsonl'

# 用于存储结果
results = []
# 用于存储 model_answer 和 COTs 数据大于 10 的问题 ID
invalid_ids = []
cnt = 0
# 读取 JSONL 文件
with open(input_file, 'r', encoding='utf-8') as infile:
    question_data = {}
 
    # 逐行读取，合并每个问题的十行数据
    for line in infile:
        data = json.loads(line, object_hook=custom_json_decoder)
        question_id = data['id']  # 获取问题的 ID
        
        # 初始化问题数据
        if question_id not in question_data:
            question_data[question_id] = {
                'query': data['query'],
                'passages': [],  # 初始化 passages 列表
                'model_answers': [],
                'ground_truth': data['ground_truth'],
                'data_type': data['data_type'],
                'COTs': []
            }

        # 将每行的 passage、model_answer 和 COT 添加到对应问题的数据中
        question_data[question_id]['passages'].append(data['passage'])  # 收集 passages
        question_data[question_id]['model_answers'].append(data['model_answer'])
        question_data[question_id]['COTs'].append(data['COT'])
    
    print(len(question_data))

    
    # 处理每个问题的数据
    for question_id, data in question_data.items():
        query = data['query']
        passages = data['passages']  # 合并后的 passages 列表
        model_answers = data['model_answers']
        ground_truth = data['ground_truth']
        datatype = data['data_type']
        COTs = data['COTs']

        if len(model_answers) > 10 or len(COTs) > 10:
            invalid_ids.append(question_id)

        correct_answers = []
        incorrect_answers = []

        if datatype in ['math_qa', 'commonsense_qa','aqua_rat']:
            # 使用准确性判断
            for model_answer, cot in zip(model_answers, COTs):
                model_answer_len = len(model_answer)
                minindex = min(model_answer_len, 15)
                choice_answer = model_answer[:minindex]
                if cnt<5:
                    print(choice_answer)
                cnt +=1
                if ground_truth.lower() in choice_answer.lower():
                    correct_answers.append((model_answer, cot))
                else:
                    incorrect_answers.append((model_answer, cot))

            # 随机从正确和错误的回答中选择一个
            chosen_answer = random.choice(correct_answers) if correct_answers else (None, None)
            rejected_answer = random.choice(incorrect_answers) if incorrect_answers else (None, None)
        
        elif datatype in ['ecqa', 'gsm8k','strategyqa', 'web_questions']:
            # 使用准确性判断
            for model_answer, cot in zip(model_answers, COTs):
                if  ground_truth.lower() in model_answer.lower():
                    correct_answers.append((model_answer, cot))
                else:
                    incorrect_answers.append((model_answer, cot))

            # 随机从正确和错误的回答中选择一个
            chosen_answer = random.choice(correct_answers) if correct_answers else (None, None)
            rejected_answer = random.choice(incorrect_answers) if incorrect_answers else (None, None)

        elif datatype in ['wiki_qa','yahoo_answers_qa','marcoqa']:
            # 使用 ROUGE 分数打分
            scored_answers = []
            for model_answer, cot in zip(model_answers, COTs):
                score = _rougel_score(model_answer, ground_truth)
                scored_answers.append((model_answer, cot, score))

            # 按照分数排序，获取最高和最低分数的回答
            scored_answers.sort(key=lambda x: x[2])  # 按分数排序 从低到高
            
            # 最高分的答案为 chosen，最低分的答案为 rejected
            chosen_answer = scored_answers[-1] if scored_answers else (None, None, None)
            rejected_answer = scored_answers[0] if scored_answers else (None, None, None)

        # 构建结果
        result_entry = {
            'id': question_id,
            'query': query,
            'model_answer': {
                'chosen': chosen_answer[0],  # chosen model answer
                'rejected': rejected_answer[0]  # rejected model answer
            },
            'COT': {
                'chosen': chosen_answer[1],  # chosen COT
                'rejected': rejected_answer[1]  # rejected COT
            },
            'passages': passages,  # 合并后的 passages 列表
            'ground_truth':ground_truth,
            'data_type': datatype
        }

        results.append(result_entry)

    print(len(results))
# 将结果写入输出文件
with open(output_file, 'w', encoding='utf-8') as outfile:
    for result in results:
        outfile.write(json.dumps(result, ensure_ascii=False) + '\n')

# 输出 model_answer 和 COTs 数据大于 10 的问题 ID
print("Question IDs with more than 10 model answers or COTs:")
for question_id in invalid_ids:
    print(question_id)