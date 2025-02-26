import torch
from transformers import (AutoModelForCausalLM, AutoTokenizer, Trainer,
                          TrainingArguments)

from datasets import Dataset
from functools import partial
import logging
from trl import DPOTrainer, KTOTrainer
import transformers
import json
from dataclasses import dataclass, field
from typing import Dict, Optional
from datasets import load_dataset, Dataset
import torch

logger = logging.getLogger(__name__)
from peft import PeftConfig, PeftModel
from template import (
    IGNORE_INDEX,
    PROMPT_DICT,
    user_tokens,
    assistant_tokens,
    pythia_user_tokens,
    pythia_assistant_tokens, RESPONSE_START_TOKEN_IDS,
)

@dataclass
class ModelArguments:
    model_name_or_path: Optional[str] = field(default="Meta-Llama-3-8B-Instruct")
    llama_style: bool = field(default=True)
    use_template: bool = field(default=True)


@dataclass
class DataArguments:
    train_data_path: str = field(
        default=None,
        metadata={"help": "Path to the training data."},
    )
    eval_data_path: str = field(
        default=None,
        metadata={"help": "Path to the test data."},
    )
    
    max_length: int = field(default=1628,metadata={"help":"Maximum all sequence length."},)
    max_prompt_length: int = field(default=1500,metadata={"help":"Maximum prompt sequence length."},)
    
    top_n: int = field(default=10,metadata={"help":"how many psg use."},)


@dataclass
class TrainingArguments(transformers.TrainingArguments):
    cache_dir: Optional[str] = field(default=None)
    optim: str = field(default="adamw_torch")
    load_lora_model : bool = field(default=True)

    use_lora: bool = field(default=True)
    output_dir : str = field(default=None)
    save_steps : int = field(default=50)
    eval_steps : int = field(default=50)
    per_device_train_batch_size: int = field(default=1)
    evaluation_strategy: str = field(default='steps')
    logging_steps : int = field(default=3)
    logging_dir : str = field(default=None)
    bf16 : bool = field(default=True)



def load_model_and_tokenizer(
    model_path: str,
    llama_style: bool,
    use_lora: bool = True,
    bf16: bool = False,
    fp16: bool = False,
    load_lora_model: bool = False,
):
    """load model and tokenizer"""


    assert not (bf16 and fp16), "bf16 or fp16, not both"
    if bf16:
        dtype = torch.bfloat16
    elif fp16:
        dtype = torch.float16
    else:
        dtype = torch.float32

    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=dtype,
        trust_remote_code=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)


    tokenizer.pad_token = tokenizer.eos_token
    if use_lora:
        from peft import LoraConfig, TaskType, get_peft_model

        if llama_style:   
            lora_config = LoraConfig(
                    task_type=TaskType.CAUSAL_LM,
                    r=8,
                    lora_alpha=32,
                    lora_dropout=0.1,
                    inference_mode=False,
                )
        else:
            lora_config = LoraConfig(
                init_lora_weights="gaussian",
                task_type=TaskType.CAUSAL_LM,
                target_modules=["q_proj", "v_proj"],

                # target_modules=['q_a_proj',
                #     'q_b_proj',
                #     'kv_a_proj_with_mqa',
                #     'kv_b_proj',
                #     'o_proj',],
                r=8,
                lora_alpha=32,
                lora_dropout=0.1,
                inference_mode=False,
            )
        model = get_peft_model(model, lora_config)
        # trainable params: 2,949,120 || all params: 3,010,652,928 || trainable%: 0.09795616002669305
        model.print_trainable_parameters()
        # model.enable_input_require_grads()  # need when using adapter

    return model, tokenizer

def preprocessing(example,args,tokenizer):
        one_item = {}
        datatype = example['data_type']
        if datatype in ['math_qa', 'commonsense_qa', 'aqua_rat', 'ecqa']:
            template = PROMPT_DICT['Mutichoice_querypassage_to_CoT']
        if datatype in ['gsm8k', 'strategyqa', 'web_questions', 'wiki_qa', 'yahoo_answers_qa', 'marcoqa']:
            template = PROMPT_DICT['QA_querypassage_to_CoT']
        query = example['query']    
        psgs = example['passages'][:args.top_n]
        psg_list = []

        for p in psgs:
            p_text = p['segment']
            if isinstance(p_text, str):
                psg_list.append(p_text)

        aug_psg = '\n'.join(psg_list)
        token_query = tokenizer([query])
        query_length = len(token_query.input_ids[0])
        
        token_aug_psg = tokenizer([aug_psg])
        token_aug_psg = token_aug_psg.input_ids[0][:args.max_prompt_length-32-query_length]
        new_aug_psg = tokenizer.decode(token_aug_psg,skip_special_tokens=True)
        
        if model_args.llama_style:

            input_data = template.format(passages=new_aug_psg, question=query)
            aug_query = [{"role": "user", "content": input_data},]
            aug_query = tokenizer.apply_chat_template(aug_query, add_generation_prompt=True, tokenize=False)
        

        one_item["prompt"] = aug_query
        one_item["chosen"] = example["model_answer"]["chosen"]
        one_item["rejected"] = example["model_answer"]["rejected"]

        return one_item

if __name__ == "__main__":
    parser = transformers.HfArgumentParser(
        (ModelArguments, DataArguments, TrainingArguments)
    )

    model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO
    )
    logger.info("Training/evaluation parameters %s", training_args)
    logger.info("MODEL parameters %s", model_args)
    logger.info("DATA parameters %s", data_args)



    model, tokenizer = load_model_and_tokenizer(
        model_path=model_args.model_name_or_path,
        llama_style = model_args.llama_style,
        use_lora=training_args.use_lora,
        bf16=training_args.bf16,
        fp16=training_args.fp16,
        load_lora_model =training_args.load_lora_model
    )
    
    partial_preprocess = partial(preprocessing,args=data_args,tokenizer=tokenizer)

    train_dataset = load_dataset("json", data_files=data_args.train_data_path,split="train")
    train_dataset = train_dataset.map(partial_preprocess)

    eval_dataset = load_dataset("json", data_files=data_args.eval_data_path,split="train")
    eval_dataset = eval_dataset.map(partial_preprocess)

    dpo_trainer = DPOTrainer(
        model,
        ref_model=None,
        args=training_args,
        beta=0.1,
        train_dataset=train_dataset,
        eval_dataset =eval_dataset,
        max_length = data_args.max_length,
        max_prompt_length = data_args.max_prompt_length,
        tokenizer=tokenizer,

    )
    dpo_trainer.train()
    dpo_trainer.save_model()




