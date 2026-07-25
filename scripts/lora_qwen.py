from __future__ import annotations

import argparse
import json
import math
import random
from pathlib import Path

import numpy as np
import torch
from datasets import DatasetDict, load_dataset
from peft import LoraConfig, TaskType, get_peft_model
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train Qwen2.5-0.5B-Instruct on LIMA using LoRA."
    )

    parser.add_argument(
        "--model-name",
        type=str,
        default="Qwen/Qwen2.5-0.5B-Instruct",
    )
    parser.add_argument(
        "--train-file",
        type=str,
        default="data/train_data.jsonl",
    )
    parser.add_argument(
        "--test-file",
        type=str,
        default="data/test_data.jsonl",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="outputs/sft_qwen",
    )

    parser.add_argument("--max-length", type=int, default=1024)
    parser.add_argument("--num-train-epochs", type=float, default=3.0)
    parser.add_argument("--learning-rate", type=float, default=2e-4)

    parser.add_argument("--train-batch-size", type=int, default=2)
    parser.add_argument("--eval-batch-size", type=int, default=2)
    parser.add_argument(
        "--gradient-accumulation-steps",
        type=int,
        default=8,
    )

    parser.add_argument("--warmup-ratio", type=float, default=0.03)
    parser.add_argument("--weight-decay", type=float, default=0.0)

    parser.add_argument("--logging-steps", type=int, default=10)
    parser.add_argument("--eval-steps", type=int, default=50)
    parser.add_argument("--save-steps", type=int, default=50)
    parser.add_argument("--save-total-limit", type=int, default=2)

    parser.add_argument("--lora-r", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument("--lora-dropout", type=float, default=0.05)

    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument(
        "--resume-from-checkpoint",
        type=str,
        default=None,
    )

    parser.add_argument(
        "--max-train-samples",
        type=int,
        default=None,
    )
    parser.add_argument(
        "--max-eval-samples",
        type=int,
        default=None,
    )

    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def validate_files(args: argparse.Namespace) -> None:
    train_path = Path(args.train_file)
    test_path = Path(args.test_file)

    if not train_path.exists():
        raise FileNotFoundError(
            f"Training file does not exist: {train_path}"
        )

    if not test_path.exists():
        raise FileNotFoundError(
            f"Test file does not exist: {test_path}"
        )


def load_lima_dataset(args: argparse.Namespace) -> DatasetDict:
    dataset = load_dataset(
        "json",
        data_files={
            "train": args.train_file,
            "test": args.test_file,
        },
    )

    for split_name in dataset.keys():
        split = dataset[split_name]

        if "messages" not in split.column_names:
            raise KeyError(
                f"The '{split_name}' split has columns "
                f"{split.column_names}, but a 'messages' column "
                f"is required."
            )

    if args.max_train_samples is not None:
        train_count = min(
            args.max_train_samples,
            len(dataset["train"]),
        )
        dataset["train"] = dataset["train"].select(
            range(train_count)
        )

    if args.max_eval_samples is not None:
        eval_count = min(
            args.max_eval_samples,
            len(dataset["test"]),
        )
        dataset["test"] = dataset["test"].select(
            range(eval_count)
        )

    return dataset


def load_tokenizer(model_name: str):
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        use_fast=True,
    )

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    tokenizer.padding_side = "right"

    return tokenizer


def tokenize_dataset(
    dataset: DatasetDict,
    tokenizer,
    max_length: int,
) -> DatasetDict:
    def preprocess_batch(batch):
        formatted_texts = []

        for messages in batch["messages"]:
            formatted_text = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=False,
            )

            formatted_texts.append(formatted_text)

        tokenized_batch = tokenizer(
            formatted_texts,
            truncation=True,
            max_length=max_length,
            padding=False,
        )

        return tokenized_batch

    tokenized_dataset = dataset.map(
        preprocess_batch,
        batched=True,
        remove_columns=dataset["train"].column_names,
        desc="Formatting and tokenizing dataset",
    )

    return tokenized_dataset


def load_lora_model(args: argparse.Namespace):
    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA is not available. Run this script through "
            "Modal or on a machine with a CUDA GPU."
        )

    bf16_supported = torch.cuda.is_bf16_supported()

    if bf16_supported:
        model_dtype = torch.bfloat16
    else:
        model_dtype = torch.float16

    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        dtype=model_dtype,
    )

    model.config.use_cache = False
    model.gradient_checkpointing_enable()

    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        bias="none",
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
    )

    model = get_peft_model(
        model,
        lora_config,
    )

    model.print_trainable_parameters()

    return model, bf16_supported


def create_training_arguments(
    args: argparse.Namespace,
    bf16_supported: bool,
) -> TrainingArguments:
    training_args = TrainingArguments(
        output_dir=args.output_dir,

        num_train_epochs=args.num_train_epochs,
        learning_rate=args.learning_rate,

        per_device_train_batch_size=args.train_batch_size,
        per_device_eval_batch_size=args.eval_batch_size,
        gradient_accumulation_steps=(
            args.gradient_accumulation_steps
        ),

        warmup_ratio=args.warmup_ratio,
        weight_decay=args.weight_decay,

        logging_strategy="steps",
        logging_steps=args.logging_steps,

        eval_strategy="steps",
        eval_steps=args.eval_steps,

        save_strategy="steps",
        save_steps=args.save_steps,
        save_total_limit=args.save_total_limit,

        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,

        bf16=bf16_supported,
        fp16=not bf16_supported,

        gradient_checkpointing=True,

        optim="adamw_torch",
        lr_scheduler_type="cosine",

        report_to="none",
        remove_unused_columns=False,

        seed=args.seed,
        data_seed=args.seed,
    )

    return training_args


def save_training_summary(
    args: argparse.Namespace,
    trainer: Trainer,
    train_size: int,
    eval_size: int,
) -> None:
    output_dir = Path(args.output_dir)
    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    evaluation_metrics = trainer.evaluate()

    eval_loss = float(
        evaluation_metrics["eval_loss"]
    )

    if eval_loss < 20:
        perplexity = math.exp(eval_loss)
    else:
        perplexity = float("inf")

    effective_batch_size = (
        args.train_batch_size
        * args.gradient_accumulation_steps
    )

    run_summary = {
        "model_name": args.model_name,
        "train_examples": train_size,
        "eval_examples": eval_size,
        "max_length": args.max_length,
        "num_train_epochs": args.num_train_epochs,
        "learning_rate": args.learning_rate,
        "train_batch_size": args.train_batch_size,
        "gradient_accumulation_steps": (
            args.gradient_accumulation_steps
        ),
        "effective_batch_size": effective_batch_size,
        "lora": {
            "rank": args.lora_r,
            "alpha": args.lora_alpha,
            "dropout": args.lora_dropout,
        },
        "eval_loss": eval_loss,
        "perplexity": perplexity,
    }

    summary_path = output_dir / "run_summary.json"

    with summary_path.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            run_summary,
            file,
            indent=2,
        )

    history_path = output_dir / "log_history.json"

    with history_path.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            trainer.state.log_history,
            file,
            indent=2,
        )

    print()
    print("Final evaluation")
    print("-" * 80)
    print(f"Evaluation loss: {eval_loss:.4f}")
    print(f"Perplexity: {perplexity:.4f}")


def main() -> None:
    args = parse_args()

    set_seed(args.seed)
    validate_files(args)

    print("Training configuration")
    print("-" * 100)
    print(f"Model: {args.model_name}")
    print(f"Train file: {args.train_file}")
    print(f"Test file: {args.test_file}")
    print(f"Output directory: {args.output_dir}")

    if torch.cuda.is_available():
        print(
            "CUDA device:",
            torch.cuda.get_device_name(0),
        )
    else:
        print("CUDA device: None")

    dataset = load_lima_dataset(args)

    print()
    print("Dataset")
    print("-" * 100)
    print("Training examples:", len(dataset["train"]))
    print("Evaluation examples:", len(dataset["test"]))
    print("Columns:", dataset["train"].column_names)

    tokenizer = load_tokenizer(
        args.model_name
    )

    tokenized_dataset = tokenize_dataset(
        dataset=dataset,
        tokenizer=tokenizer,
        max_length=args.max_length,
    )

    model, bf16_supported = load_lora_model(
        args
    )

    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,
    )

    training_arguments = create_training_arguments(
        args=args,
        bf16_supported=bf16_supported,
    )

    trainer = Trainer(
        model=model,
        args=training_arguments,
        train_dataset=tokenized_dataset["train"],
        eval_dataset=tokenized_dataset["test"],
        data_collator=data_collator,
    )

    trainer.train(
        resume_from_checkpoint=(
            args.resume_from_checkpoint
        )
    )

    final_adapter_directory = (
        Path(args.output_dir) / "final_adapter"
    )

    trainer.model.save_pretrained(
        final_adapter_directory
    )

    tokenizer.save_pretrained(
        final_adapter_directory
    )

    trainer.save_state()

    save_training_summary(
        args=args,
        trainer=trainer,
        train_size=len(
            tokenized_dataset["train"]
        ),
        eval_size=len(
            tokenized_dataset["test"]
        ),
    )

    print()
    print("Training complete")
    print("-" * 100)
    print(
        "Final LoRA adapter saved to:",
        final_adapter_directory,
    )


if __name__ == "__main__":
    main()
