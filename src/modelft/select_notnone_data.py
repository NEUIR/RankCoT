import json

input_file = 'llama3ft_dpodata.jsonl'
output_file = 'llama3ft_dpodata_notnone.jsonl'

with open(input_file, 'r', encoding='utf-8') as infile, open(output_file, 'w', encoding='utf-8') as outfile:
    for line in infile:
        
        data = json.loads(line)
        model_answer = data.get('model_answer', {})
        cot = data.get('COT', {})

     
        model_chosen = model_answer.get('chosen')
        model_rejected = model_answer.get('rejected')
        cot_chosen = cot.get('chosen')
        cot_rejected = cot.get('rejected')

       
        if not model_chosen or not model_rejected or not cot_chosen or not cot_rejected:
            continue

        outfile.write(json.dumps(data) + '\n')
