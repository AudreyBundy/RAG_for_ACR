import os
import json
import torch
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
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments,
)


def train(
        base_model: str = "/autodl-fs/data/vicuna-7b-v1.5",  # 基础模型路径
        data_path: str = "../dataDependence/train.jsonl",  # 训练数据路径
        output_dir: str = "./vicuna-finetunedv5",  # 输出模型保存路径
        batch_size: int = 4,  # 调整批次大小
        micro_batch_size: int = 1,
        num_epochs: int = 3,
        learning_rate: float = 5e-5,  # 降低学习率
        cutoff_len: int = 512,  # 减小最大输入长度
        val_set_size: int = 100,
        lora_r: int = 16,  # 提高 LoRA rank
        lora_alpha: int = 32,
        lora_dropout: float = 0.1,  # 增加 Dropout
        lora_target_modules: List[str] = ["q_proj", "v_proj"],  # 微调的模块
        resume_from_checkpoint: str = None,
        data_fraction: float = 0.5,  # 使用数据的比例
):
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128"  # 动态分配显存

    print(f"Loading base model and tokenizer from {base_model}")
    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        torch_dtype=torch.float16,
        device_map="auto"
    )
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

    # 计算数据集长度
    num_samples = len(data["train"])
    selected_num = int(num_samples * data_fraction)
    print(f"Total samples: {num_samples}, using {data_fraction * 100}%: {selected_num}")

    # 打乱数据集并选择相应比例的数据
    data_subset = data["train"].shuffle(seed=42).select(range(selected_num))

    # 数据预处理函数
    def preprocess_function(example):
        input_or = {
            "old_code": example["old_code"],
            "similarFix": example["similarFix"]
        }
        input_or = json.dumps(input_or, ensure_ascii=False)
        input_text = f"Refer to similarFix to analyze the old_code below and identify any issues.\n{input_or}"

        # 修改 target_text 的键值映射为正确字段
        target_text = {
            "defect_code": example['defect_code'],  # 确保字段名正确
            "bugType": example['bugType'],  # 确保字段名正确
            "fix": example['fix']  # 确保字段名正确
        }
        target_text = json.dumps(target_text, ensure_ascii=False)

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
        inputs["labels"] = [-100] * len(inputs["input_ids"])
        inputs["labels"][:len(targets["input_ids"])] = targets["input_ids"]
        return inputs

    tokenized_datasets = data_subset.map(
        preprocess_function,
        batched=False,
        remove_columns=data_subset.column_names
    )

    # 验证数据标签有效性（仅部分验证）
    for idx, example in enumerate(tokenized_datasets.select(range(10))):
        print(f"Sample {idx}:")
        print("Input IDs:", example["input_ids"])
        print("Attention Mask:", example["attention_mask"])
        print("Labels:", example["labels"])
        print("---")
        assert isinstance(example["labels"], list), f"Invalid labels at index {idx}: {example['labels']}"

    # 划分训练集和验证集
    train_val = tokenized_datasets.train_test_split(test_size=val_set_size)
    train_data = train_val["train"]
    val_data = train_val["test"]

    # 优化器和学习率调度器
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    total_steps = (len(train_data) // (batch_size // micro_batch_size)) * num_epochs
    warmup_steps = int(0.1 * total_steps)  # 10% 的 warmup
    lr_scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps
    )

    # 数据整理器
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,  # 因为是因果语言模型
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
        evaluation_strategy="epoch",  # 在每个 epoch 后验证
        save_strategy="epoch",
        save_total_limit=2,
        load_best_model_at_end=True,
        fp16=True,  # 启用半精度训练
        gradient_checkpointing=False,  # 可关闭以避免显存问题
        report_to="none",
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
