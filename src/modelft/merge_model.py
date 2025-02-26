import argparse
import json
import os

import numpy as np
import torch
from peft import PeftConfig, PeftModel
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

model_name_or_path = "trained_model_output_dir/checkpoint/checkpoint-[lowest eval loss step]"

config = PeftConfig.from_pretrained(model_name_or_path)
base_tokenizer = AutoTokenizer.from_pretrained(config.base_model_name_or_path)
model = AutoModelForCausalLM.from_pretrained(
    config.base_model_name_or_path,
    torch_dtype=torch.bfloat16,
    trust_remote_code=True,
    )
model = PeftModel.from_pretrained(model, model_name_or_path)
model = model.merge_and_unload()
model.save_pretrained("mergemodel/mergemodel_output_dir")
base_tokenizer.save_pretrained("mergemodel/mergemodel_output_dir")

print("---------------------------")