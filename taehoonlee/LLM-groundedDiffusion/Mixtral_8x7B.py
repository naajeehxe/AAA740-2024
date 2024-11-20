import torch
import os
import huggingface_hub

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

#! notice #!
# this model works with transformers==4.41.2 and 4.4

#gated model so login first
huggingface_hub.login()

# load model
model_id = "mistralai/Mixtral-8x7B-Instruct-v0.1"
tokenizer = AutoTokenizer.from_pretrained(model_id)

model = AutoModelForCausalLM.from_pretrained(model_id, device_map="auto")

# prompt generation
template = templatev0_1
template_version = '0.1'
message = "In an indoor scene, a blue cube directly above a red cube with a vase on the left of them."

print(template)

prompt = f"{template}### User: {message}\n\n### Assistant:\n"

inputs = tokenizer.apply_chat_template(prompt, return_tensors="pt").to("cuda")
outputs = model.generate(inputs, max_new_tokens=20)
print(output)


# messages = [
#     {"role": "user", "content": "What is your favourite condiment?"},
#     {"role": "assistant", "content": "Well, I'm quite partial to a good squeeze of fresh lemon juice. It adds just the right amount of zesty flavour to whatever I'm cooking up in the kitchen!"},
#     {"role": "user", "content": "Do you have mayonnaise recipes?"}
# ]

# inputs = tokenizer.apply_chat_template(messages, return_tensors="pt").to("cuda")

# outputs = model.generate(inputs, max_new_tokens=20)
# print(tokenizer.decode(outputs[0], skip_special_tokens=True))
