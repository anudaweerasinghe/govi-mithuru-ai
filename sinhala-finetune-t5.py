from datasets import load_dataset
from huggingface_hub import login
import wandb
import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, Seq2SeqTrainingArguments, DataCollatorForSeq2Seq, Seq2SeqTrainer

import os

model_name = "sinhala-aya-mt5"
base_model_name = "google/mt5-large"
dataset_name = "CohereForAI/aya_dataset"

os.environ["WANDB_PROJECT"] = "sinhala-llm"
os.environ["WANDB_LOG_MODEL"] = "false"  # don't log model checkpoints

login(token=os.environ['HF_TOKEN'])
wandb.login(key=os.environ['WANDB_KEY'])

def preprocess_function(examples):
  model_inputs =  tokenizer(examples["inputs"], max_length=1024, truncation=True)
  labels = tokenizer(examples["targets"])

  model_inputs["labels"] = labels["input_ids"]

  return model_inputs

dataset = load_dataset(dataset_name, split="all")
sinhala_dataset = dataset.filter(lambda x: x['language_code'] == 'sin')

base_model = AutoModelForSeq2SeqLM.from_pretrained(base_model_name, torch_dtype=torch.bfloat16)
tokenizer = AutoTokenizer.from_pretrained(base_model_name)
data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer)

tokenizer.pad_token = tokenizer.eos_token

tokenized_sinhala_dataset = sinhala_dataset.map(preprocess_function)

training_args = Seq2SeqTrainingArguments(
  output_dir="mt5-model-out",
  num_train_epochs=3,
  learning_rate=3e-4,
  per_device_train_batch_size=1,
  per_device_eval_batch_size=1,
  fp16=False,
  bf16=True,
  logging_dir="distilled-model/logs",
  logging_strategy="steps",
  logging_steps=10,
  save_strategy="epoch",
  # save_steps=25000,
  push_to_hub=True,
  hub_strategy="every_save",
  hub_model_id=model_name,
  report_to="wandb",
  lr_scheduler_type="constant",
  gradient_accumulation_steps=64,
  gradient_checkpointing=True,
  optim="adafactor"
)

trainer = Seq2SeqTrainer(
  model=base_model,
  args=training_args,
  train_dataset=tokenized_sinhala_dataset,
  tokenizer=tokenizer,
  data_collator=data_collator,
)

trainer.train()


