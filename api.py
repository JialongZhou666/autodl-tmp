from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "1"

def generate_response(prompt):
    input_str = f'<|start_header_id|>user<|end_header_id|>\n\n{prompt}<|eot_id|>'
    encodings = tokenizer(input_str, return_tensors='pt').to('cuda')
    
    generated_ids = model.generate(
        **encodings,
        max_new_tokens=256,
        do_sample=True, 
        top_p=0.7,
        temperature=0.3,
        repetition_penalty=1.2,
        pad_token_id=tokenizer.eos_token_id,  # 设置pad_token_id
        eos_token_id=tokenizer.eos_token_id   # 设置eos_token_id
    )
    
    response = tokenizer.decode(generated_ids[0][encodings['input_ids'].shape[1]:])
    return response.strip().replace('<|eot_id|>', "").replace('<|start_header_id|>assistant<|end_header_id|>\n\n', '').strip()

model_name = 'LLM-Research/Meta-Llama-3-8B-Instruct'
tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token    # 设置pad_token
model = AutoModelForCausalLM.from_pretrained(model_name, device_map={"": 0}, torch_dtype=torch.bfloat16)

while True:
    prompt = input("You: ")
    if prompt == "exit": break
    print("Assistant:", generate_response(prompt))