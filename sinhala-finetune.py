from datasets import load_dataset
from huggingface_hub import login
import wandb
import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, TrainingArguments, DataCollatorForLanguageModeling

from trl import SFTTrainer
import os

model_name = "sinhala-aya-mt5"
base_model_name = "google/mt5-xl"
dataset_name = "CohereForAI/aya_dataset"

os.environ["WANDB_PROJECT"] = "sinhala-llm"
os.environ["WANDB_LOG_MODEL"] = "false"  # don't log model checkpoints

login(token=os.environ['HF_TOKEN'])
wandb.login(key=os.environ['WANDB_KEY'])

def format_instructions(sample):
  outputs = []

  for i in range(len(sample['targets'])):
    outputs.append(f"""
{sample['inputs'][i]}

{sample['targets'][i]}
                   """)
    
  return outputs

dataset = load_dataset(dataset_name, split="train")
sinhala_dataset = dataset.filter(lambda x: x['language'] == 'si')

base_model = AutoModelForSeq2SeqLM.from_pretrained(base_model_name, torch_dtype=torch.bfloat16)
tokenizer = AutoTokenizer.from_pretrained(base_model_name)
data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

tokenizer.pad_token = tokenizer.eos_token

training_args = TrainingArguments(
  output_dir="model-out",
  num_train_epochs=1,
  learning_rate=3e-4,
  per_device_train_batch_size=1,
  per_device_eval_batch_size=1,
  fp16=False,
  bf16=True,
  logging_dir="distilled-model/logs",
  logging_strategy="steps",
  logging_steps=100,
  save_strategy="steps",
  save_steps=25000,
  push_to_hub=True,
  hub_strategy="every_save",
  hub_model_id=model_name,
  report_to="wandb",
  lr_scheduler_type="constant",
  gradient_accumulation_steps=64,
  gradient_checkpointing=True,
)

trainer = SFTTrainer(
  model=base_model,
  args=training_args,
  formatting_func=format_instructions,
  packing=False,
)

trainer.train()


