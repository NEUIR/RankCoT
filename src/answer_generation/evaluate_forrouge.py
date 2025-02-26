import json
from rouge import Rouge

def _rougel_score(prediction, ground_truth):
    rouge = Rouge()
    try:
        scores = rouge.get_scores(prediction, ground_truth, avg=True)
        return scores["rouge-l"]["f"]
    except ValueError:  # Hypothesis is empty
        return 0.0


def calculate_average_rougel(jsonl_file):
    total_score = 0.0
    num_questions = 0

    with open(jsonl_file, "r", encoding="utf-8") as file:
        for line in file:
            data = json.loads(line)
            model_answer = data["model_answer"]  
            ground_truths = data["ground_truth"]  
            
            max_score = max(_rougel_score(model_answer, gt) for gt in ground_truths)
            
            total_score += max_score
            num_questions += 1

    
    return total_score / num_questions if num_questions > 0 else 0.0


jsonl_file = "marco_queryCoT_to_answer.jsonl"  
average_rougel = calculate_average_rougel(jsonl_file)
print(f"Average ROUGE-L score: {average_rougel:.4f}")