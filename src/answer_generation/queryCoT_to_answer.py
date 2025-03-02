import os 
import json 
import argparse 
import numpy as np
from tqdm import tqdm 
from vllm import LLM, SamplingParams
from template import PROMPT_DICT
from transformers import AutoTokenizer
from torch.utils.data import Dataset, DataLoader

def custom_json_decoder(obj):
    if 'id' in obj:
        obj['id'] = str(obj['id'])
    return obj

class llmDataset(Dataset):
    def __init__(self,data,tokenizer):
        self.data = data
        self.tokenizer = tokenizer
        
    def process_prompt(self, item):
        id=item['id']
        query = item['query']
        passages = item['passages']
        ground_truth = item['ground_truth']
        cot = item['model_output']
        template = PROMPT_DICT['QA_queryCoT_to_answer']
        template = template.format(question=query, COT=cot)
        messages = [
            {"role": "user", "content": template},
        ]
        input_prompt = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        item['input_prompt'] = input_prompt
        # item['ground_truth'] = ground_truth

        return item
    
            
    def __getitem__(self, index):
        item = self.data[index]       
        item = self.process_prompt(item)

        if index == 0:
            print(item)
       
        return item
    
    def __len__(self):
        return len(self.data)
    
    def Collactor (self, batch):
        
        id = [f['id'] for f in batch]
        query = [f['query'] for f in batch]
        passages = [f['passages'] for f in batch]
        ground_truth = [f['ground_truth'] for f in batch]
        COT = [f['model_output'] for f in batch]
        input_prompt = [f['input_prompt'] for f in batch]
        
        return{'id':id,
               'query':query,
               'passages':passages,
               'COT':COT,
               'ground_truth':ground_truth,
               'input_prompt': input_prompt
        }
    
def inference (args):
    # Load data from the JSONL file
    with open(args.data_path, 'r') as file:
        data = [json.loads(line, object_hook=custom_json_decoder) for line in file]

    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    dataset = llmDataset(data,tokenizer)
    dataloader = DataLoader(dataset=dataset, batch_size=60, collate_fn= dataset.Collactor)
    params_dict = {
            "n": 1,
            "best_of": 1,
            "presence_penalty": 1.0,
            "frequency_penalty": 0.0,
            "temperature": 1.0,
            "top_p": 1.0,
            "top_k": 1,
            "use_beam_search": False,
            "length_penalty": 1,
            "early_stopping": False,
            "stop": None,
            "stop_token_ids": None,
            "ignore_eos": False,
            "max_tokens": 32,
            "logprobs": None,
            "prompt_logprobs": None,
            "skip_special_tokens": True,
        }
    sampling_params = SamplingParams(**params_dict)
    # cuda_num = len(os.getenv('CUDA_VISIBLE_DEVICES').split(','))
    cuda_num = 1
    llm = LLM(
        model=args.model_path, 
        tensor_parallel_size=cuda_num, 
        dtype='bfloat16',
        trust_remote_code=True,
        gpu_memory_utilization=0.9
    )
    output_data = []
    
    for batch in tqdm(dataloader):
        input_prompt = batch['input_prompt']
        outputs: list =llm.generate(input_prompt, sampling_params)
        cleaned_outputs = [output.outputs[0].text for output in outputs]
        maxindex = len(batch['id'])
        for index in range(maxindex):
            id=batch['id'][index]
            query = batch['query'][index]
            passages = batch['passages'][index]
            cot = batch['COT'][index]
            ground_truth = batch['ground_truth'][index]
            model_answer = cleaned_outputs[index] 
            output_item = {
                "id":id,
                "query": query,
                "passages": passages,
                "COT":cot,
                "model_answer": model_answer,
                "ground_truth":ground_truth
                }
            output_data.append(output_item)

    with open(args.output_name, 'w') as outfile:
        for item in output_data:
            json.dump(item, outfile)
            outfile.write('\n')

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str,default="Meta-Llama-3-8B-Instruct")
    parser.add_argument('--data_path',type=str, default="nq_querypassage_to_CoT.jsonl")
    parser.add_argument('--output_name',type=str,default="nq_queryCoT_to_answer.jsonl")
    args = parser.parse_args ()
    inference(args)
