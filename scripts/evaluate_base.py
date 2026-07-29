from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import torch
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate the base Qwen2.5-0.5B-Instruct model "
            "on the held-out LIMA evaluation split."
        )
    )

    parser.add_argument(
        "--model-name",
        type=str,
        default="Qwen/Qwen2.5-0.5B-Instruct",
    )

    parser.add_argument(
        "--eval-file",
        type=str,
        default="data/eval_data_130.jsonl",
    )

    parser.add_argument(
        "--output-file",
        type=str,
        default="outputs/evaluation/base_metrics.json",
    )

    parser.add_argument(
        "--max-length",
        type=int,
        default=1024,
    )

    parser.add_argument(
        "--eval-batch-size",
        type=int,
        default=2,
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
    )

    return parser.parse_args()


def validate_file(eval_file: str) -> None:
    eval_path = Path(eval_file)

    if not eval_path.exists():
        raise FileNotFoundError(
            f"Evaluation file does not exist: {eval_path}"
        )


def load_eval_dataset(eval_file: str):
    dataset = load_dataset(
        "json",
        data_files={
            "eval": eval_file,
        },
    )

    eval_dataset = dataset["eval"]

    if "messages" not in eval_dataset.column_names:
        raise KeyError(
            "The evaluation dataset must contain a "
            "'messages' column."
        )

    if len(eval_dataset) != 130:
        raise ValueError(
            f"Expected 130 evaluation examples, "
            f"found {len(eval_dataset)}."
        )

    return eval_dataset


def validate_conversations(eval_dataset) -> None:
    for index, example in enumerate(eval_dataset):
        messages = example["messages"]

        has_user_message = any(
            message["role"] == "user"
            for message in messages
        )

        has_assistant_message = any(
            message["role"] == "assistant"
            for message in messages
        )

        if not has_user_message:
            raise ValueError(
                f"Evaluation example {index} does not "
                "contain a user message."
            )

        if not has_assistant_message:
            raise ValueError(
                f"Evaluation example {index} does not "
                "contain an assistant response."
            )


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
    eval_dataset,
    tokenizer,
    max_length: int,
):
    def preprocess_batch(batch):
        formatted_texts = []

        for messages in batch["messages"]:
            formatted_text = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=False,
            )

            formatted_texts.append(formatted_text)

        return tokenizer(
            formatted_texts,
            truncation=True,
            max_length=max_length,
            padding=False,
        )

    tokenized_dataset = eval_dataset.map(
        preprocess_batch,
        batched=True,
        remove_columns=eval_dataset.column_names,
        desc="Formatting and tokenizing evaluation dataset",
    )

    return tokenized_dataset


def load_base_model(model_name: str):
    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA is not available. Run this script "
            "through Modal or another CUDA machine."
        )

    if torch.cuda.is_bf16_supported():
        model_dtype = torch.bfloat16
        bf16_supported = True
    else:
        model_dtype = torch.float16
        bf16_supported = False

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        dtype=model_dtype,
    )

    model.eval()

    return model, bf16_supported


def evaluate_model(
    model,
    tokenized_dataset,
    tokenizer,
    eval_batch_size: int,
    seed: int,
    bf16_supported: bool,
) -> dict:
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,
    )

    evaluation_arguments = TrainingArguments(
        output_dir="/tmp/base_evaluation",
        per_device_eval_batch_size=eval_batch_size,
        bf16=bf16_supported,
        fp16=not bf16_supported,
        report_to="none",
        remove_unused_columns=False,
        seed=seed,
    )

    trainer = Trainer(
        model=model,
        args=evaluation_arguments,
        eval_dataset=tokenized_dataset,
        data_collator=data_collator,
    )

    metrics = trainer.evaluate()

    eval_loss = float(metrics["eval_loss"])

    if eval_loss < 20:
        perplexity = math.exp(eval_loss)
    else:
        perplexity = float("inf")

    return {
        "eval_loss": eval_loss,
        "perplexity": perplexity,
        "eval_runtime": float(metrics["eval_runtime"]),
        "eval_samples_per_second": float(
            metrics["eval_samples_per_second"]
        ),
        "eval_steps_per_second": float(
            metrics["eval_steps_per_second"]
        ),
    }


def save_metrics(
    args: argparse.Namespace,
    metrics: dict,
    number_of_examples: int,
) -> None:
    output_path = Path(args.output_file)

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    result = {
        "model_type": "base",
        "model_name": args.model_name,
        "eval_file": args.eval_file,
        "eval_examples": number_of_examples,
        "max_length": args.max_length,
        "eval_batch_size": args.eval_batch_size,
        "loss_masking": "full_sequence_except_padding",
        **metrics,
    }

    with output_path.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            result,
            file,
            indent=2,
        )

    print()
    print("Base-model evaluation")
    print("-" * 80)
    print(f"Evaluation examples: {number_of_examples}")
    print(f"Evaluation loss: {metrics['eval_loss']:.4f}")
    print(f"Perplexity: {metrics['perplexity']:.4f}")
    print(f"Saved metrics to: {output_path}")


def main() -> None:
    args = parse_args()

    validate_file(args.eval_file)

    print("Evaluation configuration")
    print("-" * 100)
    print(f"Model: {args.model_name}")
    print(f"Evaluation file: {args.eval_file}")
    print(f"Output file: {args.output_file}")
    print(f"Maximum sequence length: {args.max_length}")
    print(f"Evaluation batch size: {args.eval_batch_size}")

    eval_dataset = load_eval_dataset(
        args.eval_file
    )

    validate_conversations(
        eval_dataset
    )

    print()
    print("Dataset")
    print("-" * 100)
    print("Evaluation examples:", len(eval_dataset))
    print("Columns:", eval_dataset.column_names)

    tokenizer = load_tokenizer(
        args.model_name
    )

    tokenized_dataset = tokenize_dataset(
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        max_length=args.max_length,
    )

    model, bf16_supported = load_base_model(
        args.model_name
    )

    metrics = evaluate_model(
        model=model,
        tokenized_dataset=tokenized_dataset,
        tokenizer=tokenizer,
        eval_batch_size=args.eval_batch_size,
        seed=args.seed,
        bf16_supported=bf16_supported,
    )

    save_metrics(
        args=args,
        metrics=metrics,
        number_of_examples=len(tokenized_dataset),
    )


if __name__ == "__main__":
    main()
    