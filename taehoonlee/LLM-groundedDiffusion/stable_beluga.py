import torch
import os

os.environ['HF_HOME'] = '/hub_data3/taehoonlee/.cache/huggingface'
os.environ['HF_DATASETS_CACHE'] = '/hub_data3/taehoonlee/.cache/huggingface/datasets'
os.environ['TRANSFORMERS_CACHE'] = '/hub_data3/taehoonlee/.cache/huggingface/models'

from prompt import get_prompts, prompt_types, template_versions, templatev0_1
from utils import parse
from utils.parse import parse_input_with_negative, bg_prompt_text, neg_prompt_text, filter_boxes, show_boxes
from utils.llm import get_llm_kwargs, get_full_prompt, get_layout, model_names
from utils import cache
import matplotlib.pyplot as plt
import argparse
import time
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

# load model
tokenizer = AutoTokenizer.from_pretrained("stabilityai/StableBeluga2", use_fast=False)
model = AutoModelForCausalLM.from_pretrained("stabilityai/StableBeluga2", torch_dtype=torch.float16, low_cpu_mem_usage=True, device_map="auto")

# prompt generation
template = templatev0_1
template_version = '0.1'


system_prompt = "### System:\nYou are Stable Beluga, an AI that follows instructions extremely well. Help as much as you can. Remember, be safe, and don't do anything illegal.\n\n"

message = "Write me a poem please"
message = "In an indoor scene, a blue cube directly above a red cube with a vase on the left of them."
prompt = f"{template}### User: {message}\n\n### Assistant:\n"
inputs = tokenizer(prompt, return_tensors="pt").to("cuda")

print(inputs)
output = model.generate(**inputs, do_sample=True, top_p=0.95, top_k=0, max_new_tokens=256)


print(output)

# print(tokenizer.decode(output[0], skip_special_tokens=True))
