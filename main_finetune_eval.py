import os
import torch
import re
import wandb
import gc

from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments,BitsAndBytesConfig, Trainer, DataCollatorWithPadding, DataCollatorForSeq2Seq, get_scheduler
from datasets import load_dataset
from trl import SFTTrainer, DataCollatorForCompletionOnlyLM
from peft import AutoPeftModelForCausalLM, LoraConfig, get_peft_model, prepare_model_for_kbit_training, TaskType, PrefixTuningConfig
from datasets import load_from_disk
from transformers import LlamaForCausalLM, LlamaTokenizer
# from utils import find_all_linear_names, print_trainable_parameters
from torch.utils.data import DataLoader
from tqdm import tqdm

from accelerate import Accelerator

import torch
import bitsandbytes as bnb
import fire
from datetime import timedelta
import json
from datasets import Dataset
# from function_sam_trainer import FSDPFunctionalSAMTrainer
# from sam_trainer import FSDPSAMTrainer

from optimizer_torch import GaLoreAdamW, GaLoreAdamW8bit, GaLoreAdafactor, APOLLO, QAPOLLO, SAM, FunctionalSAM
 



def eval(
        output_dir="",
        logging_dir="",
        model_name ="",
        dataset_name: str="omini_math",
        use_samples: int=-1,
        batch_size: int = 8,
        cutoff_len: int = 800,
        bf16=True,
        local_rank: int=-1,
        gradient_checkpoint: bool=False,
):
    if output_dir=="":
        output_dir=f"{model_name}/eval" 

    if dataset_name == "MATH":
        dataset = load_dataset("json", data_files={"train": "MATH/train.jsonl", "test": "MATH/test.jsonl"})
        train_data = dataset["train"]
        test_data = dataset["test"]
        def add_prompt_column(example):
            example["prompt"] = f"Question: {example['question']}\nAnswer: "
            example["response"] = example["solution"]
            return example

        train_data =train_data.map(add_prompt_column)
        test_data = test_data.map(add_prompt_column)


    elif dataset_name == "GSM":
        dataset  = load_dataset("openai/gsm8k", "main")
        train_data = dataset["train"]
        test_data = dataset["test"]
        def add_prompt_column(example):
            example["prompt"] = f"Question: {example['question']}\nAnswer: "
            example["response"] = example["answer"]
            return example
        train_data =train_data.map(add_prompt_column)
        test_data = test_data.map(add_prompt_column)
    elif dataset_name == "MATH_instruct":
        # template= "Below is an instruction that describes a task.\nWrite a response that appropriately completes the request.\n\n### Instruction:\n{}\n\n### Response:\n"
        # dataset=load_dataset("TIGER-Lab/MathInstruct", split="train")
        # dataset = dataset.shuffle(seed=42)
        # train_num = int(len(dataset)*0.95)
        # train_data = dataset.select(range(train_num))
        # test_data = dataset.select(range(train_num, len(dataset)))
        # def add_prompt_column(example):
        #     example["prompt"] = template.format(example["instruction"])
        #     example["response"] = example["output"]
        #     return example
        # train_data =train_data.map(add_prompt_column)
        # test_data = test_data.map(add_prompt_column)
       
        train_data = load_dataset("json", data_files="data/Math_Instruct/train_data.json")["train"]
        test_data = load_dataset("json", data_files="data/Math_Instruct/test_data.json")["train"]

    else:
        raise ValueError(f"Dataset {dataset_name} not supported")
    
    if use_samples>0:
        train_data=train_data.select(range(use_samples))

    
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    # max_memory = {i: f"{int(torch.cuda.get_device_properties(i).total_memory * 0.1 / 1e6)}MB" for i in range(torch.cuda.device_count())}
    if torch.cuda.device_count() > 1:
        # max_memory = {i: f"3000MB" for i in range(torch.cuda.device_count())}
        # max_memory = {0: f"3000MB", 1:"10000MB"}
        model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto")
    else:
        model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto")
    tokenizer.padding_side="right"
    tokenizer.pad_token=tokenizer.eos_token


    # model.print_trainable_parameters()

    if gradient_checkpoint:
        model.gradient_checkpointing_enable()

    if bf16:
        model = model.to(torch.bfloat16)





    def tokenizer_function(examples):
        prompt_len = len(tokenizer.encode(examples["prompt"], add_special_tokens=False, max_length=cutoff_len, truncation=True))
        inputs = tokenizer(examples["prompt"]+examples["response"]+tokenizer.eos_token, add_special_tokens=False, max_length=cutoff_len, truncation=True, return_tensors="pt")
        inputs = {k: v.squeeze(0) for k, v in inputs.items()}
        inputs["labels"] = inputs["input_ids"].clone()
        inputs["labels"][:prompt_len] = -100
        return inputs
        


    tokenized_dataset = train_data.map(tokenizer_function, remove_columns=train_data.column_names)
    test_tokenized_dataset = test_data.map(tokenizer_function, remove_columns=test_data.column_names)
    # tokenized_dataset = train_data.map(tokenizer_function)
    # test_tokenized_dataset = test_data.map(tokenizer_function)

    # data_collator=DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model, padding=True, max_length=cutoff_len)

    # train_loader = DataLoader(tokenized_dataset, batch_size=batch_size, shuffle=False,collate_fn=data_collator)
    # test_loader = DataLoader(test_tokenized_dataset, batch_size=batch_size, shuffle=False,collate_fn=data_collator)



    train_datas=[]
    test_datas=[]
    train_loss=0
    eval_loss=0

    model.eval()

    # train_loader_tqdm = tqdm(train_loader, desc=f"Training", leave=True)
    # for batch in train_loader_tqdm:
    #     labels = batch["labels"].to(model.device)
    #     batch = {k: v.to(model.device) for k, v in batch.items() if k != "labels"}
    #     logits = model(**batch).logits
    #     shift_logits = logits[..., :-1, :].contiguous()
    #     shift_labels = labels[..., 1:].contiguous()
    #     loss_fct = torch.nn.CrossEntropyLoss()
    #     bs=shift_logits.size(0)
    #     for i in range(bs):
    #         loss = loss_fct(shift_logits[i].view(-1, shift_logits.size(-1)), shift_labels[i].view(-1))
    #         train_loss+=loss.detach().cpu().item()
    #         train_datas.append({"text":tokenizer.decode(batch["input_ids"][i], skip_special_tokens=True), "loss":loss.detach().cpu().item()})
    # train_loss = train_loss/len(train_data)


    os.makedirs(output_dir, exist_ok=True)
    with open(os.path.join(output_dir, "train_eval.json"), "w") as f:
        for data in tqdm(tokenized_dataset):
            batch = {"input_ids":torch.LongTensor(data["input_ids"]).unsqueeze(0).to(model.device), "attention_mask":torch.LongTensor(data["attention_mask"]).unsqueeze(0).to(model.device), "labels":torch.LongTensor(data["labels"]).unsqueeze(0).to(model.device)}
            loss = model(**batch).loss
            train_loss+=loss.detach().cpu().item()
            output={"text":tokenizer.decode(batch["input_ids"][0], skip_special_tokens=True), "loss":loss.detach().cpu().item()}
            f.write(json.dumps(output)+"\n")
            train_datas.append(output)
    train_loss = train_loss/len(train_datas)

    print(f"Train Loss: {train_loss}")

   
    # with open(os.path.join(output_dir, "train_eval.json"), "w") as f:
    #     json.dump(train_datas, f)
    

    # test_loader_tqdm = tqdm(test_loader, desc=f"Testing", leave=True)
    # for batch in test_loader_tqdm:
    #     labels = batch["labels"].to(model.device)
    #     batch = {k: v.to(model.device) for k, v in batch.items() if k != "labels"}
    #     logits = model(**batch).logits
    #     shift_logits = logits[..., :-1, :].contiguous()
    #     shift_labels = labels[..., 1:].contiguous()
    #     loss_fct = torch.nn.CrossEntropyLoss()
    #     bs=shift_logits.size(0)
    #     for i in range(bs):
    #         loss = loss_fct(shift_logits[i].view(-1, shift_logits.size(-1)), shift_labels[i].view(-1))
    #         eval_loss+=loss.detach().cpu().item()
    #         test_datas.append({"text":tokenizer.decode(batch["input_ids"][i], skip_special_tokens=True), "loss":loss.detach().cpu().item()})
    with open(os.path.join(output_dir, "test_eval.json"), "w") as f:
        for data in tqdm(test_tokenized_dataset):
            batch = {"input_ids":torch.LongTensor(data["input_ids"]).unsqueeze(0).to(model.device), "attention_mask":torch.LongTensor(data["attention_mask"]).unsqueeze(0).to(model.device), "labels":torch.LongTensor(data["labels"]).unsqueeze(0).to(model.device)}
            loss = model(**batch).loss
            eval_loss+=loss.detach().cpu().item()
            output={"text":tokenizer.decode(batch["input_ids"][0], skip_special_tokens=True), "loss":loss.detach().cpu().item()}
            f.write(json.dumps(output)+"\n")
            test_datas.append(output)
    eval_loss = eval_loss/len(test_datas)
    # with open(os.path.join(output_dir, "test_eval.json"), "w") as f:
    #     json.dump(test_datas, f)



    print(f"Eval Loss: {eval_loss}")

if __name__ == "__main__":
    fire.Fire(eval)