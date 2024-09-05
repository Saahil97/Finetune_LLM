import pandas as pd

import torch
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
    pipeline
)

from transformers import LlamaTokenizer, LlamaTokenizerFast
from peft import LoraConfig
from trl import SFTTrainer

huggingface_token = "############################################"

def generate_prompt(x):
    output_texts = []

    for i in range(len(x['question'])):
        sys_prompt = """You are a subject matter expert (SME) in the Sports Entertainment field, 
        You will be given a question related to WWE or WWE superstars, Your job is to use the Options, 
        jumble them up, because usually correct answer is the first option.
        Also specify which is the right choice.
        """
        question = f'\nQuestion: {x["question"][i]}\nOptions:\n' \
                   f'a: {x["Correct Answer"][i]}\n' \
                   f'b: {x["Wrong Answer 1"][i]}\n' \
                   f'c: {x["Wrong Answer 2"][i]}\n' \
                   f'd: {x["Wrong Answer 3"][i]}\n' \
                   f'Correct: {x["Correct Answer"][i]}\n'
        
        prompt = f"""<s> [INST] <<SYS>> {sys_prompt} <</SYS>> {question} </s>"""
        output_texts.append(prompt)

    return output_texts

dataset = load_dataset('csv', data_files={'train': ['wwe_mcq_dataset1_utf8.csv']})

base_model_name = "mistralai/Mistral-7B-Instruct-v0.1"
refined_model = "mistral-7b-wwetuned"
output_dir = "./results_wwetuned"
epochs = 10

try:
    llama_tokenizer = LlamaTokenizerFast.from_pretrained(
        base_model_name, 
        trust_remote_code=True,
        use_auth_token=huggingface_token
    )
except Exception as e:
    print("Fast tokenizer failed, trying non-fast version...")
    from transformers import LlamaTokenizer
    llama_tokenizer = LlamaTokenizer.from_pretrained(
        base_model_name, 
        trust_remote_code=True,
        use_auth_token=huggingface_token
    )

llama_tokenizer.pad_token_id = 18610
llama_tokenizer.padding_side = "right"

# Activate 4-bit precision base model loading
use_4bit = False
use_8_bit = False

# Compute dtype for 4-bit base models
bnb_4bit_compute_dtype = "bfloat16"


# Load tokenizer and model with QLoRA configuration
compute_dtype = getattr(torch, bnb_4bit_compute_dtype)

# Check GPU compatibility with bfloat16
if compute_dtype == torch.float16 and use_4bit:
    major, _ = torch.cuda.get_device_capability()
    if major >= 8:
        print("=" * 80)
        print("Your GPU supports bfloat16: accelerate training with bf16=True")
        print("=" * 80)
        
quant_config = BitsAndBytesConfig(
    load_in_4bit=False,
    # load_in_8bit=use_8_bit,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=compute_dtype,
    bnb_4bit_use_double_quant=False,
)

print(f"Loading model...")
# Model
base_model = AutoModelForCausalLM.from_pretrained(
    base_model_name,
    # pretrained_model_name_or_path = base_model_name,
    # quantization_config=quant_config,
    torch_dtype=torch.float16,
    device_map="auto",
    use_auth_token=huggingface_token
    # device_map={"": }
)

base_model.config.use_cache = False
base_model.config.pretraining_tp = 1

lora_target_modules = [
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "gate_proj",
    "down_proj",
    "up_proj",
]

# LoRA Config
peft_parameters = LoraConfig(
    lora_alpha=128,
    lora_dropout=0.1,
    target_modules=lora_target_modules,
    r=64,
    bias="none",
    task_type="CAUSAL_LM"
)

# Training Params
train_params = TrainingArguments(
    output_dir=output_dir,
    num_train_epochs=epochs,
    per_device_train_batch_size=2,
    gradient_accumulation_steps=2,
    optim="paged_adamw_32bit",
    save_steps=250,
    # save_total_limit=10,
    # load_best_model_at_end=True,
    logging_steps=100,
    learning_rate=2e-4,
    weight_decay=0.001,
    fp16=True,
    bf16=False,
    max_grad_norm=1.0,
    max_steps=-1,
    warmup_ratio=0.03,
    group_by_length=True,
    lr_scheduler_type="cosine",
    report_to="tensorboard"
)

# Trainer
fine_tuning = SFTTrainer(
    model=base_model,
    train_dataset=dataset['train'],
    # eval_dataset=dataset['val'],
    peft_config=peft_parameters,
    max_seq_length=1024,
    tokenizer=llama_tokenizer,
    formatting_func=generate_prompt,
    args=train_params,
)

print(f"Starting Training...")

# Training
fine_tuning.train()

# Save Model
fine_tuning.model.save_pretrained(refined_model)
fine_tuning.tokenizer.save_pretrained(refined_model)
