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
 



def train_sft(
        output_dir="",
        logging_dir="",
        model_name ="",
        dataset_name: str="omini_math",
        use_samples: int=-1,
        batch_size: int = 8,
        eval_batch_size: int = 8,
        micro_batch_size: int = -1,
        num_train_epochs: int = 10,
        learning_rate: float = 5e-6,
        cutoff_len: int = 1000,
        bf16=True,
        local_rank: int=-1,
        gradient_checkpoint: bool=False,
        optimizer_name: str ="adam",
        sam_rho: float=0.05,
        sam_precond: bool=False,
        eval_steps: int=5000,
        save_steps: int=5000,
):
    global_rank=0
    if micro_batch_size==-1:
        micro_batch_size=batch_size
    # global_rank = int(os.environ['RANK'])
    # local_rank = int(os.environ["LOCAL_RANK"])

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
        template= "Below is an instruction that describes a task.\nWrite a response that appropriately completes the request.\n\n### Instruction:\n{}\n\n### Response:\n"
        dataset=load_dataset("TIGER-Lab/MathInstruct", split="train")
        dataset = dataset.shuffle(seed=42)
        train_num = int(len(dataset)*0.95)
        train_data = dataset.select(range(train_num))
        test_data = dataset.select(range(train_num, len(dataset)))
        def add_prompt_column(example):
            example["prompt"] = template.format(example["instruction"])
            example["response"] = example["output"]
            return example
        train_data =train_data.map(add_prompt_column)
        test_data = test_data.map(add_prompt_column)
        # train_data.to_json("data/Math_Instruct/train_data.json")
        # test_data.to_json("data/Math_Instruct/test_data.json")
        # train_data = load_dataset("json", data_files="data/Math_Instruct/train_data.json")["train"]
        # test_data = load_dataset("json", data_files="data/Math_Instruct/test_data.json")["train"]
        
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

    data_collator=DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model, padding=True, max_length=cutoff_len)

    train_loader = DataLoader(tokenized_dataset, batch_size=batch_size, shuffle=True,collate_fn=data_collator)
    test_loader = DataLoader(test_tokenized_dataset, batch_size=eval_batch_size, shuffle=False,collate_fn=data_collator)

    run_name = f"{dataset_name}_{os.path.basename(output_dir)}"
    
    wandb.init(project="SAM", name=run_name)

    trainable_params = [p for p in model.parameters() if p.requires_grad]
    if optimizer_name.lower() == "adamw":
        optimizer = torch.optim.AdamW(trainable_params, lr=learning_rate, weight_decay=0)
    elif optimizer_name.lower() == "self_adamw":
        optimizer = GaLoreAdamW(trainable_params, lr=learning_rate, weight_decay=0)
    elif optimizer_name.lower() == "sam":
        base_optimizer = torch.optim.AdamW
        optimizer = SAM(trainable_params, base_optimizer, lr=learning_rate, rho=sam_rho, precond=sam_precond)
    elif optimizer_name.lower() == "function_sam":
        base_optimizer = torch.optim.AdamW
        optimizer = FunctionalSAM(trainable_params, base_optimizer, lr=learning_rate, rho=sam_rho, precond=sam_precond)

    scheduler = get_scheduler(
        name="cosine",
        optimizer=optimizer,
        num_warmup_steps=1000,
        num_training_steps=len(train_loader)*num_train_epochs,
    )
    global_micro_batch_size = micro_batch_size
    global_step=0
    for epoch in range(num_train_epochs):
        model.train()
        train_loader_tqdm = tqdm(train_loader, desc=f"Epoch {epoch+1} - Training", leave=True)
        for batch in train_loader_tqdm:
            global_step+=1
            model.train()
            labels = batch["labels"].to(model.device)
            batch = {k: v.to(model.device) for k, v in batch.items() if k != "labels"}

            if optimizer_name.lower() in ["function_sam"] and labels.shape[1]>500:
                micro_batch_size = 1
            else:
                micro_batch_size = global_micro_batch_size
            
            bs=batch["input_ids"].shape[0]
            accumulated_steps = (bs+micro_batch_size-1)//micro_batch_size
            accumulated_loss = 0
            if optimizer_name.lower() in ["sam"]:
                for i in range(0, bs, micro_batch_size):
                    micro_batch = {k: v[i:i+micro_batch_size] for k, v in batch.items()}
                    micro_labels = labels[i:i+micro_batch_size]
                    outputs = model(**micro_batch, labels=micro_labels)
                    loss = outputs.loss / accumulated_steps
                    accumulated_loss+=loss.item()
                    loss.backward()
                optimizer.first_step(zero_grad=True)
                for i in range(0, bs, micro_batch_size):
                    micro_batch = {k: v[i:i+micro_batch_size] for k, v in batch.items()}
                    micro_labels = labels[i:i+micro_batch_size]
                    outputs = model(**micro_batch, labels=micro_labels)
                    loss = outputs.loss / accumulated_steps
                    loss.backward()
                optimizer.second_step(zero_grad=True)
            elif optimizer_name.lower() in ["function_sam"]:
                def network_fn(params, batch):
                    return torch.func.functional_call(model, params, (), batch).logits
                dL_dlogits = []
                for i in range(0, bs, micro_batch_size):
                    micro_batch = {k: v[i:i+micro_batch_size] for k, v in batch.items()}
                    micro_labels = labels[i:i+micro_batch_size]
                    outputs = model(**micro_batch, labels=micro_labels)
                    loss = outputs.loss / accumulated_steps
                    logits = outputs.logits
                    accumulated_loss+=loss.item()
                    loss.backward()
                    outputs = model(**micro_batch, labels=micro_labels)
                    logits = outputs.logits
                    loss = outputs.loss / accumulated_steps
                    with torch.no_grad():
                        dL_dlogit = torch.autograd.grad(
                            loss,       
                            logits,     
                            retain_graph=False,
                            allow_unused=True
                        )[0].detach()
                        dL_dlogits.append(dL_dlogit)
                    del loss, logits, outputs
                    gc.collect()
                
                optimizer.first_step(zero_grad=True)
                grads = None
                perturbed_params = {name: param.detach() for name, param in model.named_parameters()}
                # grad_names = [name for name, param in model.named_parameters() if param.requires_grad]
                # perturbed_params = [p for group in optimizer.param_groups for p in group["params"] if p.requires_grad]
                for i in range(0, bs, micro_batch_size):
                    with torch.no_grad():
                        micro_batch = {k: v[i:i+micro_batch_size] for k, v in batch.items()}
                        micro_labels = labels[i:i+micro_batch_size]
                        # with torch.enable_grad():
                        #     outputs = model(**micro_batch, labels=micro_labels)
                        #     perturbed_logits = outputs.logits
                        #     perturbed_loss = outputs.loss
                        # perturbed_loss.backward(create_graph=True)
                        # dL_dlogit = dL_dlogits[i//micro_batch_size]
                        # all_params = [p for group in optimizer.param_groups for p in group["params"] if p.grad is not None]
                        # grad = torch.autograd.grad(
                        #     perturbed_logits, 
                        #     all_params,       
                        #     grad_outputs=dL_dlogit,  
                        #     retain_graph=False
                        # )
                        dF_dtheta_fn = torch.func.vjp(lambda theta: network_fn(theta, micro_batch), perturbed_params)[1]
                        for k, v in micro_batch.items():
                            v.detach()
                            del k, v
                        del micro_batch
                        vjp_grads= dF_dtheta_fn(dL_dlogits[i//micro_batch_size])[0]
                        # grad = [grad[n].detach().clone() for n in grad_names]
                        # if grads is None:
                        #     grads = [g for g in grad]
                        # else:
                        #     grads = [g1+g2 for g1, g2 in zip(grads, grad)]
                        # del grad
                        model_param_dict = dict(model.named_parameters())
                        for name, grad in vjp_grads.items():
                            if grad is not None:
                                param = model_param_dict.get(name)
                                if param is not None and grad is not None:
                                    if param.grad is None:
                                        # param.grad = torch.zeros_like(param, device=model.device)
                                        param.grad = grad.detach() 
                                    else:
                                        param.grad = param.grad + grad.detach() 
                            del name, grad
                        del vjp_grads
                        gc.collect()
                optimizer.final_step(grads)

                del grads, dL_dlogits
                gc.collect()        
            else:
                for i in range(0, bs, micro_batch_size):
                    micro_batch = {k: v[i:i+micro_batch_size] for k, v in batch.items()}
                    micro_labels = labels[i:i+micro_batch_size]
                    outputs = model(**micro_batch, labels=micro_labels)
                    loss = outputs.loss / accumulated_steps
                    accumulated_loss+=loss.item()
                    loss.backward()
                optimizer.step()

            train_loader_tqdm.set_description(f"Epoch {epoch+1} - Train Loss: {accumulated_loss:.4f}")
            del batch, labels
            gc.collect()
            

            if global_rank == 0:
                wandb.log({
                    "train_loss": accumulated_loss,
                    "lr": optimizer.param_groups[0]["lr"],
                    },
                    step=global_step,
                )
            scheduler.step()
            optimizer.zero_grad()
            
            if global_step%eval_steps==0:
                model.eval()
                eval_loss = 0
                test_loader_tqdm = tqdm(test_loader, desc=f"Step {global_step} - Testing", leave=True)
                for batch in test_loader_tqdm:
                    with torch.no_grad():
                        labels = batch["labels"].to(model.device)
                        batch = {k: v.to(model.device) for k, v in batch.items() if k != "labels"}
                        outputs = model(**batch, labels=labels)
                        loss = outputs.loss
                        eval_loss+=loss.item()
                        test_loader_tqdm.set_description(f"Epoch {epoch+1} - Eval Loss: {loss.item():.4f}")
                        # wandb.log({"test_loss": loss.item()})
                eval_loss/=len(test_loader)
                if global_rank == 0:
                    wandb.log({"eval_loss": eval_loss}, step=global_step)
            if global_rank == 0 and global_step%save_steps==0:
                save_dir = f"{output_dir}/step-{global_step}"
                os.makedirs(save_dir, exist_ok=True)
                model.save_pretrained(save_dir)
                tokenizer.save_pretrained(save_dir)
        



if __name__ == "__main__":
    fire.Fire(train_sft)