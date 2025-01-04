import os
import sys
import json
import torch
import transformers
from typing import List
from datasets import load_dataset
from peft import (
    LoraConfig,
    get_peft_model,
    set_peft_model_state_dict,
)
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    get_linear_schedule_with_warmup,
    DataCollatorForSeq2Seq,
    Trainer,
    TrainingArguments,
)
from torch.distributed import init_process_group

def train(
    base_model: str = "/autodl-fs/data/vicuna-7b-v1.5",  # 基础模型路径
    data_path: str = "../dataDependence/train.jsonl",  # 训练数据路径
    output_dir: str = "./vicuna-finetuned",  # 输出模型保存路径
    batch_size: int = 8,
    micro_batch_size: int = 4,
    num_epochs: int = 3,
    learning_rate: float = 5e-5,
    cutoff_len: int = 1000,  # 最大输入长度
    val_set_size: int = 2000,
    lora_r: int = 8,
    lora_alpha: int = 16,
    lora_dropout: float = 0.05,
    lora_target_modules: List[str] = ["q_proj", "v_proj"],  # 微调的模块
    resume_from_checkpoint: str = None,
):
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128"  # 动态分配显存
    print(f"Loading base model and tokenizer from {base_model}")
    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        torch_dtype=torch.float16,
        device_map="auto"
    )
    model.gradient_checkpointing_enable()
    tokenizer = AutoTokenizer.from_pretrained(base_model, use_fast=False)


    # 配置 LoRA 微调
    config = LoraConfig(
        r=lora_r,
        lora_alpha=lora_alpha,
        target_modules=lora_target_modules,
        lora_dropout=lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, config)

    if resume_from_checkpoint:
        checkpoint_name = os.path.join(resume_from_checkpoint, "pytorch_model.bin")
        if os.path.exists(checkpoint_name):
            print(f"Resuming from checkpoint: {checkpoint_name}")
            checkpoint_weights = torch.load(checkpoint_name)
            set_peft_model_state_dict(model, checkpoint_weights)

    # 加载数据集
    print(f"Loading dataset from {data_path}")
    data = load_dataset("json", data_files={"train": data_path})

    # 数据预处理函数
    def preprocess_function(example, i = 0):
        input_or = {
            "old_code": example["old_code"],
            "similarFix": example["similarFix"]
        }
        input_or = json.dumps(input_or, ensure_ascii=False)
        input_text = f"Refer to similarFix to analyze the old_code below and identify any issues.{input_or}"
        target_text = {
            "defect_code": {example['defect_code']},
            "bugType": {example['fix']},
            "fix": {example['bugType']}
        }
        target_text = json.dumps(target_text, ensure_ascii=False)
        # print(target_text)
        inputs = tokenizer(
            input_text,
            max_length=cutoff_len,
            truncation=True,
            padding="max_length",
        )
        targets = tokenizer(
            target_text,
            max_length=cutoff_len,
            truncation=True,
            padding="max_length",
        )
        inputs["labels"] = targets["input_ids"]
        # print(f"Labels: {inputs['labels']}")
        return inputs

    tokenized_datasets = data["train"].map(preprocess_function, batched=False)
    for idx, example in enumerate(tokenized_datasets):
        if not isinstance(example["labels"], list):
            print(f"Invalid labels at index {idx}: {example['labels']}")

    # 划分训练集和验证集
    if val_set_size > 0:
        train_val = tokenized_datasets.train_test_split(test_size=val_set_size)
        train_data = train_val["train"]
        val_data = train_val["test"]
    else:
        train_data = tokenized_datasets
        val_data = None

    # 优化器和学习率调度器
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    num_training_steps = (len(train_data) // batch_size) * num_epochs
    lr_scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=100, num_training_steps=num_training_steps
    )

    # 数据整理器
    data_collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        padding=True,
        pad_to_multiple_of=8,
        return_tensors="pt",
    )

    # 配置训练参数
    training_args = TrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=micro_batch_size,
        gradient_accumulation_steps=batch_size // micro_batch_size,
        num_train_epochs=num_epochs,
        learning_rate=learning_rate,
        logging_dir="./logs",
        logging_steps=10,
        # evaluation_strategy="steps" if val_set_size > 0 else "no",
        evaluation_strategy="no",
        # eval_steps=20 if val_set_size > 0 else None,
        save_strategy="steps",  # 改为按步保存
        save_steps=100,  # 每100步保存一次
        save_total_limit=3,
        # load_best_model_at_end=True if val_set_size > 0 else False,
        load_best_model_at_end = False,
        fp16=True if torch.cuda.is_available() else False,  # 启用半精度训练
        bf16=False,  # 禁用 bf16
        gradient_checkpointing=True,  # 禁用梯度检查点以加快训练
        report_to="none",  # 关闭 wandb 等日志服务
    )

    # 创建 Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_data,
        eval_dataset=val_data,
        tokenizer=tokenizer,
        data_collator=data_collator,
        optimizers=(optimizer, lr_scheduler),
    )

    # 开始训练
    print("Starting training...")
    trainer.train(resume_from_checkpoint=resume_from_checkpoint)

    # 保存模型
    print(f"Saving model to {output_dir}")
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)


if __name__ == "__main__":
    train()
