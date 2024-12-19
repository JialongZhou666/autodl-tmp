from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import os
import uvicorn

app = FastAPI()

# 初始化模型
os.environ["CUDA_VISIBLE_DEVICES"] = "3"
model_name = 'LLM-Research/Meta-Llama-3-8B-Instruct'
tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
model = AutoModelForCausalLM.from_pretrained(model_name, device_map={"": 0}, torch_dtype=torch.bfloat16)

class Query(BaseModel):
    prompt: str

@app.post("/v1/chat/completions")
async def generate(query: Query):
    try:
        input_str = f'<|start_header_id|>user<|end_header_id|>\n\n{query.prompt}<|eot_id|>'
        encodings = tokenizer(input_str, return_tensors='pt').to('cuda')
        
        generated_ids = model.generate(
            **encodings,
            max_new_tokens=2048,
            do_sample=True,
            top_p=0.7,
            temperature=0.3,
            repetition_penalty=1.2,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id
        )
        
        response = tokenizer.decode(generated_ids[0][encodings['input_ids'].shape[1]:])
        response = response.strip().replace('<|eot_id|>', "").replace('<|start_header_id|>assistant<|end_header_id|>\n\n', '').strip()
        
        return {
            "choices": [{
                "message": {
                    "content": response
                }
            }]
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8001)